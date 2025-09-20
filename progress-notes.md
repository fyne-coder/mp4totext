# mp4totext – Progress Log

_Last updated: 2025-09-16 23:17 EDT_

## Objectives
- Replace Pyannote diarization with a fully offline Apple-Silicon-friendly pipeline.
- Keep Whisper transcription local, with option to switch to Faster-Whisper.
- Achieve reliable multi-speaker diarization on long MP4 recordings without external services.

## Environment & Tooling
- Virtualenv: `.venv311` (Python 3.11) — required for TensorFlow/Torchaudio compatibility.
- Core packages installed:
  - `inaSpeechSegmenter` for VAD + gender detection.
  - `tensorflow-macos 2.16.2`, `tensorflow-metal 1.2.0` (Metal acceleration / TensorFlow runtime).
  - `speechbrain` for speaker embeddings (`spkrec-ecapa-voxceleb` model cached locally).
  - `openai-whisper` (default transcription) + optional `faster-whisper` backend.
  - `moviepy`, `scikit-image`, `torchaudio`, `sklearn` for supporting features.

## Key Script Changes (`m4splitter.py`)
1. **Backend selection**
   - Added `--whisper-backend {openai,faster}`.
   - Added `--diarizer-backend {inaspeech, inaspeech-cluster, pyannote}`.
   - Added `--language` flag to suppress repeated auto-detection logs.

2. **inaSpeech diarization**
   - Patched numpy-breaking changes in Pyannote’s Viterbi helper.
   - Added channel-axis shim for TensorFlow 2.16+.
   - Suppressed inaSpeech progress output using `contextlib.redirect_stdout`.

3. **Speaker clustering (new)**
   - Merge adjacent speech VAD fragments (≤0.5 s gap); enforce minimum duration (≥1.5 s).
   - Compute ECAPA embeddings (speechbrain) on merged turns; normalize vectors.
   - When `--cluster-speakers N` provided, run KMeans (`n_init=10`); otherwise use agglomerative with cosine distance + threshold.
   - Merge consecutive segments sharing the same speaker ID (with short gap tolerance) before printing.

4. **Transcription**
   - Shared `get_transcriber` helper now accepts explicit `language` argument.
   - Transcription results for diarization are concatenated per merged turn.

## Current CLI Usage
```
source .venv311/bin/activate
python m4splitter.py INPUT.mp4 \
  --diarize \
  --model tiny \
  --whisper-backend openai \
  --diarizer-backend inaspeech-cluster \
  --cluster-speakers 2 \
  --language en \
  > diarized_output.txt
```
- Runtime for 60+ minute MP4 on M3 Max ≈ **15 minutes** (`real 14m55s`).
- Output lines look like `[SPK00 0.00-180.42] ...` with stable speaker IDs.

## Observations & Lessons Learned
- **Clustering quality** depends heavily on segment duration. Merging short VAD fragments before embedding dramatically reduces false speaker switches.
- **Explicit speaker count** resolves KMeans drift when one voice dominates the audio.
- **Suppressing logs** (`--language en` & stdout redirection) keeps transcripts clean and easier to parse.
- **inaSpeechSegmenter** still invaluable for fast local diarization but benefits from speaker embedding backends for multi-speaker scenarios.
- Processing long MP4s in a single diarization pass preserves global speaker clustering; transcription can be parallelized afterwards if needed.

## Remaining Ideas / To‑Dos
- Optional JSON export of merged segments for downstream batch processing.
- Consider voice-activity alignment tweaks (dynamic gap thresholds per silence length).
- Evaluate alternate local diarizers (SpeechBrain diarization recipe, NVIDIA NeMo) if more than 2–3 speakers are expected regularly.

## Output Artifacts
- `diarized_input_clustered_clean.txt` — latest full-run transcript with two speakers.
- `diarized_output_segment_0_clustered.txt` — smoke-test MP3 output showing correct speaker alternation.
- `progress-notes.md` (this file) — ongoing documentation of changes and learnings.
