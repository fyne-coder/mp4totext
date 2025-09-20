# mp4totext (local diarization + transcription)

Utilities for slicing media, transcribing with Whisper, and labeling speakers on an Apple Silicon MacBook.

## Environment setup (Apple Silicon)

1. Install FFmpeg (required by inaSpeechSegmenter and MoviePy):
   ```bash
   brew install ffmpeg
   ```
2. Create and activate a Python 3.11 virtual environment (the pre-created `.venv311` uses this interpreter because `inaSpeechSegmenter` is not compatible with Python 3.13):
   ```bash
   python3.11 -m venv .venv311
   source .venv311/bin/activate
   python -m pip install --upgrade pip
   ```
3. Install Python dependencies:
   ```bash
   pip install openai-whisper faster-whisper inaSpeechSegmenter tensorflow-macos tensorflow-metal moviepy
   ```
   * `tensorflow-macos`/`tensorflow-metal` are needed by inaSpeechSegmenter on Apple Silicon; the packages fall back to CPU if Metal is unavailable.
   * `faster-whisper` is optional but enabled via the CLI for faster local transcription.
   * If you encounter import errors inside the diarization stack after a numpy upgrade, rerun the virtualenv setup to ensure the runtime picks up the compatibility patches.
   * `moviepy` is only required if you use `frommoviepy.py` to split tracks by file size.
4. (Optional) If you need the Pyannote diarization backend, configure Hugging Face credentials and install its dependencies:
   ```bash
   pip install "pyannote.audio<3"
   huggingface-cli login
   export HUGGINGFACE_HUB_TOKEN=...  # or pass --hf-token on the CLI
   ```

## Usage

The main entry point is `m4splitter.py`:

```bash
python m4splitter.py INPUT_FILE [--split --duration SECONDS] \
                      [--transcribe | --diarize]
                      [--whisper-backend {openai,faster}] \
                      [--diarizer-backend {inaspeech,inaspeech-cluster,pyannote}] \
                      [--cluster-threshold 0.65] [--cluster-speakers N] \
                      [--language en]
```

Key combinations:

- `--split --duration 600` exports the first 10 minutes of audio (`<basename>_600.mp3`).
- `--transcribe` prints a Whisper transcript of the whole file. Use `--whisper-backend faster` to run via `faster-whisper`.
- `--diarize` runs local diarization with `inaSpeechSegmenter` by default, then transcribes each speaker turn with Whisper. Use `--diarizer-backend inaspeech-cluster` to assign unique local speaker IDs (via speechbrain speaker embeddings) or `--diarizer-backend pyannote` if you prefer the Pyannote pipeline (requires a Hugging Face token). Tune `--cluster-threshold` (lower → more clusters) or pass `--cluster-speakers` to force a specific number of speakers. Add `--language en` (or other ISO code) to skip Whisper’s auto-detection banner.

> **Note:** When `--diarize` is combined with `--whisper-backend faster`, the script automatically falls back to the OpenAI Whisper implementation for segment-level transcription because chunked inference with `faster-whisper` is still experimental.

## Testing

- After installing the dependencies, run a quick smoke test on the sample clip included in `file/ThriveCapital.mp4`:
  ```bash
  # Activate the 3.11 env first
  source .venv311/bin/activate
  python m4splitter.py file/ThriveCapital.mp4 --diarize --model tiny
  ```
- If you enable the Pyannote backend, verify it separately:
  ```bash
  python m4splitter.py file/ThriveCapital.mp4 --diarize --diarizer-backend pyannote --hf-token "$HUGGINGFACE_HUB_TOKEN"
  # For fully offline unique speakers, run once to cache speechbrain weights:
  python m4splitter.py file/output_segment_0.mp3 --diarize --diarizer-backend inaspeech-cluster
  ```

## Known limitations

- inaSpeechSegmenter’s gender labels (`male`, `female`, `speech`) are language-dependent; adjust filtering inside `transcribe_with_inaspeech` if you need additional classes.
- The speechbrain-based clustering backend downloads the `speechbrain/spkrec-ecapa-voxceleb` weights the first time you run it; afterwards everything stays local. Use `--cluster-threshold` to tweak how aggressively speakers are separated or `--cluster-speakers` to pin the count when you already know how many voices are present.
- `faster-whisper` chunked diarization is not enabled by default; if you need it, extend `get_transcriber` to handle numpy audio segments with explicit sampling rates.
