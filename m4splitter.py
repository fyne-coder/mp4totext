#!/usr/bin/env python3
import argparse
import contextlib
import io
import os
import subprocess
import sys
import numpy as np
import torch
import whisper

# Optional dependency used for local diarization. Imported lazily to avoid
# forcing TensorFlow install when the feature is unused.
try:
    from inaSpeechSegmenter import Segmenter  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime when diarization is requested
    Segmenter = None  # type: ignore

try:
    from speechbrain.inference.speaker import EncoderClassifier  # type: ignore
except ImportError:  # pragma: no cover - optional diarization backend
    EncoderClassifier = None  # type: ignore


def get_transcriber(backend, model_name, language=None):
    """Return a callable that transcribes audio chunks using the requested backend."""

    backend = backend.lower()
    if backend == "openai":
        model = whisper.load_model(model_name)

        def _transcribe(audio):
            result = model.transcribe(audio, verbose=False, language=language)
            return result["text"].strip()

        return _transcribe

    if backend == "faster":
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "faster-whisper is not installed. Install it with 'pip install faster-whisper'."
            ) from exc

        model = WhisperModel(model_name, compute_type="int8" if model_name != "large" else "float16")

        def _transcribe(audio):
            segments, _ = model.transcribe(audio, beam_size=5, language=language)
            return " ".join(segment.text.strip() for segment in segments if segment.text)

        return _transcribe

    raise ValueError(f"Unsupported whisper backend: {backend}")

def split_audio(input_file, duration):
    """Extract the first <duration> seconds of a media file and save as MP3."""
    """Extract the first <duration> seconds of a media file and save as MP3 using ffmpeg."""
    base, _ = os.path.splitext(input_file)
    output_file = f"{base}_{duration}.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_file, "-t", str(duration), "-q:a", "0", "-map", "a", output_file],
        check=True,
    )
    return output_file

def transcribe_audio(input_file, model_name, backend="openai", language=None):
    """Transcribe audio from a media file using the requested Whisper backend."""

    transcribe = get_transcriber(backend, model_name, language=language)
    return transcribe(input_file)

def transcribe_with_inaspeech(
    input_file,
    model_name,
    backend="openai",
    vad_engine="smn",
    detect_gender=True,
    ffmpeg_binary="ffmpeg",
    language=None,
):
    """Diarize audio locally with inaSpeechSegmenter then run Whisper."""

    if Segmenter is None:
        raise ImportError(
            "inaSpeechSegmenter is not installed. Install it with 'pip install "
            "tensorflow-macos tensorflow-metal inaSpeechSegmenter' on Apple Silicon."
        )

    segmenter = Segmenter(
        vad_engine=vad_engine,
        detect_gender=detect_gender,
        ffmpeg=ffmpeg_binary,
        batch_size=1,
    )

    # Keras >=2.16 expects a trailing channel axis. Some inaSpeechSegmenter releases
    # still feed (batch, time, features); adapt dynamically so we can stay fully local.
    def _wrap_predict(module):
        try:
            original = module.nn.predict  # type: ignore[attr-defined]

            def _predict_with_channel(batch, *args, **kwargs):
                if isinstance(batch, np.ndarray) and batch.ndim == 3:
                    batch = batch[..., np.newaxis]
                return original(batch, *args, **kwargs)

            module.nn.predict = _predict_with_channel  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover - older versions already handle this
            pass

    _wrap_predict(segmenter.vad)
    if detect_gender:
        _wrap_predict(segmenter.gender)
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        diarization = segmenter(input_file)

    transcribe = get_transcriber(backend, model_name, language=language)
    audio = whisper.load_audio(input_file)
    sample_rate = 16000

    transcript = []
    speech_labels = {"male", "female", "speech"}

    for label, start, end in diarization:
        if label not in speech_labels:
            continue

        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        chunk = audio[start_sample:end_sample]

        if chunk.size == 0:
            continue

        text = transcribe(chunk)
        if not text:
            continue

        speaker = label if label in {"male", "female"} else "speech"
        transcript.append(f"[{speaker} {start:.2f}-{end:.2f}] {text}")

    return "\n".join(transcript)


def transcribe_with_inaspeech_cluster(
    input_file,
    model_name,
    backend="openai",
    vad_engine="smn",
    ffmpeg_binary="ffmpeg",
    threshold=0.65,
    speaker_count=None,
    language=None,
):
    """Cluster inaSpeech segments with speaker embeddings for unique labels."""

    if Segmenter is None:
        raise ImportError(
            "inaSpeechSegmenter is not installed. Install it with 'pip install "
            "tensorflow-macos tensorflow-metal inaSpeechSegmenter'."
        )

    if EncoderClassifier is None:
        raise ImportError(
            "speechbrain is required for clustering backend. Install it with 'pip install speechbrain'."
        )

    segmenter = Segmenter(vad_engine=vad_engine, detect_gender=False, ffmpeg=ffmpeg_binary, batch_size=1)

    def _wrap_predict(module):
        try:
            original = module.nn.predict  # type: ignore[attr-defined]

            def _predict_with_channel(batch, *args, **kwargs):
                if isinstance(batch, np.ndarray) and batch.ndim == 3:
                    batch = batch[..., np.newaxis]
                return original(batch, *args, **kwargs)

            module.nn.predict = _predict_with_channel  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover - older versions already handle this
            pass

    _wrap_predict(segmenter.vad)

    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        diarization = [seg for seg in segmenter(input_file) if seg[0] == "speech"]
    if not diarization:
        return ""

    transcribe = get_transcriber(backend, model_name, language=language)
    audio = whisper.load_audio(input_file)
    sample_rate = 16000

    # Load speaker embedding model (runs locally once weights cached).
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"},
    )

    segments = []
    current_start = diarization[0][1]
    current_end = diarization[0][2]

    max_gap = 0.5  # seconds
    min_duration = 1.5  # seconds

    for _, start, end in diarization[1:]:
        if start - current_end <= max_gap:
            current_end = end
        else:
            segments.append((current_start, current_end))
            current_start, current_end = start, end
    segments.append((current_start, current_end))

    refined_segments = []
    for start, end in segments:
        duration = end - start
        if duration >= min_duration:
            refined_segments.append((start, end))
        elif refined_segments:
            refined_segments[-1] = (refined_segments[-1][0], end)
        else:
            refined_segments.append((start, end))

    embeddings = []
    chunks = []
    for start, end in refined_segments:
        start_sample = max(int(start * sample_rate), 0)
        end_sample = min(int(end * sample_rate), len(audio))
        chunk = audio[start_sample:end_sample]
        if chunk.size == 0:
            continue
        tensor = torch.from_numpy(chunk).unsqueeze(0)
        emb = encoder.encode_batch(tensor)
        emb_np = emb.squeeze().cpu().numpy()
        norm = np.linalg.norm(emb_np)
        if norm > 0:
            emb_np = emb_np / norm
        embeddings.append(emb_np)
        chunks.append((start, end, chunk))

    if not embeddings:
        return ""

    if len(embeddings) == 1:
        cluster_labels = [0]
    else:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.cluster import KMeans

        embeddings_array = np.stack(embeddings)

        if speaker_count is not None:
            if speaker_count < 1:
                raise ValueError("--cluster-speakers must be >= 1 when provided.")
            max_clusters = len(embeddings_array)
            if speaker_count > max_clusters:
                print(
                    "Requested --cluster-speakers="
                    f"{speaker_count} exceeds available diarized segments ({max_clusters}). "
                    f"Capping to {max_clusters}.",
                    file=sys.stderr,
                )
                speaker_count = max_clusters
            if speaker_count == 1:
                cluster_labels = np.zeros(len(embeddings_array), dtype=int)
            else:
                kmeans = KMeans(n_clusters=speaker_count, n_init=10, random_state=0)
                cluster_labels = kmeans.fit_predict(embeddings_array)
        else:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                metric="cosine",
                linkage="average",
                distance_threshold=threshold,
            )
            try:
                cluster_labels = clustering.fit_predict(embeddings_array)
            except TypeError:  # Older sklearn versions use 'affinity'
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    affinity="cosine",
                    linkage="average",
                    distance_threshold=threshold,
                )
                cluster_labels = clustering.fit_predict(embeddings_array)

    transcript_lines = []
    previous_label = None
    merged_start = None
    merged_end = None
    merged_text = []

    for (start, end, chunk), label in zip(chunks, cluster_labels):
        text = transcribe(chunk)
        if not text:
            continue

        if previous_label is None:
            previous_label = label
            merged_start = start
            merged_end = end
            merged_text = [text]
            continue

        if label == previous_label and start - merged_end <= max_gap:
            merged_end = end
            merged_text.append(text)
        else:
            transcript_lines.append(
                f"[SPK{previous_label:02d} {merged_start:.2f}-{merged_end:.2f}] {' '.join(merged_text)}"
            )
            previous_label = label
            merged_start = start
            merged_end = end
            merged_text = [text]

    if previous_label is not None and merged_text:
        transcript_lines.append(
            f"[SPK{previous_label:02d} {merged_start:.2f}-{merged_end:.2f}] {' '.join(merged_text)}"
        )

    return "\n".join(transcript_lines)

def transcribe_with_pyannote(input_file, model_name, backend="openai", hf_token=None, language=None):
    """Diarize audio using Pyannote (requires HF token and online download)."""
    from pyannote.audio import Pipeline

    token = hf_token or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    auth = token if token else True
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization", use_auth_token=auth
    )
    transcribe = get_transcriber(backend, model_name, language=language)
    audio = whisper.load_audio(input_file)
    sample_rate = 16000
    transcript = []
    diarization = pipeline({"audio": input_file})
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_sample = int(turn.start * sample_rate)
        end_sample = int(turn.end * sample_rate)
        chunk = audio[start_sample:end_sample]
        text = transcribe(chunk)
        if not text:
            continue
        transcript.append(f"[{speaker} {turn.start:.2f}-{turn.end:.2f}] {text}")
    return "\n".join(transcript)

def main():
    parser = argparse.ArgumentParser(description="Split audio and/or transcribe media files.")
    parser.add_argument("input", help="Input media file (e.g., .mp3, .mp4)")
    parser.add_argument("--split", action="store_true", help="Extract first segment of specified duration.")
    parser.add_argument("--duration", type=int, default=600, help="Duration in seconds for splitting.")
    parser.add_argument("--transcribe", action="store_true", help="Transcribe audio to text.")
    parser.add_argument("--diarize", action="store_true", help="Perform speaker diarization and label speech segments.")
    parser.add_argument(
        "--diarizer-backend",
        choices=["inaspeech", "inaspeech-cluster", "pyannote"],
        default="inaspeech",
        help="Select diarization backend. 'inaspeech' uses gender labels, 'inaspeech-cluster' "
        "adds local speaker clustering, and 'pyannote' requires Hugging Face auth.",
    )
    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.65,
        help="Cosine distance threshold for inaSpeech clustering backend.",
    )
    parser.add_argument(
        "--cluster-speakers",
        type=int,
        help="Optional fixed number of speakers for the inaSpeech clustering backend.",
    )
    parser.add_argument(
        "--whisper-backend",
        choices=["openai", "faster"],
        default="openai",
        help="Choose Whisper implementation. 'faster' uses faster-whisper for optimized local inference.",
    )
    parser.add_argument("--model", default="small", help="Whisper model name (tiny, base, small, medium, large).")
    parser.add_argument(
        "--language",
        help="Explicit language code to skip Whisper auto-detection (prevents repeated 'Detected language' logs).",
    )
    parser.add_argument("--hf-token", help="Hugging Face token for pyannote models (or set HUGGINGFACE_HUB_TOKEN env var)")
    args = parser.parse_args()

    if not args.split and not args.transcribe and not args.diarize:
        parser.error("At least one of --split, --transcribe, or --diarize must be specified.")

    if args.transcribe and args.diarize:
        parser.error("Use either --transcribe or --diarize, not both.")

    if args.split:
        output_file = split_audio(args.input, args.duration)
        print(f"Exported segment to {output_file}")

    if args.transcribe:
        transcript = transcribe_audio(
            args.input,
            args.model,
            backend=args.whisper_backend,
            language=args.language,
        )
        print("Transcript:")
        print(transcript)

    if args.diarize:
        if args.whisper_backend == "faster":
            print(
                "Warning: faster-whisper backend is experimental for diarization chunks; "
                "falling back to OpenAI Whisper for segment transcription.",
                flush=True,
            )
            diarization_transcriber = "openai"
        else:
            diarization_transcriber = args.whisper_backend

        if args.diarizer_backend == "inaspeech":
            transcript = transcribe_with_inaspeech(
                args.input,
                args.model,
                backend=diarization_transcriber,
                language=args.language,
            )
        elif args.diarizer_backend == "inaspeech-cluster":
            transcript = transcribe_with_inaspeech_cluster(
                args.input,
                args.model,
                backend=diarization_transcriber,
                threshold=args.cluster_threshold,
                speaker_count=args.cluster_speakers,
                language=args.language,
            )
        else:
            transcript = transcribe_with_pyannote(
                args.input,
                args.model,
                backend=diarization_transcriber,
                hf_token=args.hf_token,
                language=args.language,
            )
        print("Transcript:")
        print(transcript)

try:
    from pyannote.algorithms.utils import viterbi as _viterbi  # type: ignore

    if not getattr(_viterbi, "_mp4totext_numpy_patch", False):
        def _patched_stack(arr, counts):
            tiles = [np.tile(e, (c, 1)) for e, c in zip(arr.T, counts)]
            if not tiles:
                return np.empty_like(arr)
            return np.vstack(tiles).T

        _viterbi._update_emission = lambda emission, consecutive: _patched_stack(emission, consecutive)  # type: ignore[attr-defined]
        _viterbi._update_constraint = lambda constraint, consecutive: _patched_stack(constraint, consecutive)  # type: ignore[attr-defined]
        _viterbi._mp4totext_numpy_patch = True  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - best effort patching
    pass

if __name__ == "__main__":
    main()
