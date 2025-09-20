from moviepy.editor import VideoFileClip

def splice_audio(input_file, max_size_MB=25, audio_format="mp3"):
    # Load the video file
    video = VideoFileClip(input_file)
    
    # Extract the audio
    audio = video.audio

    # Get audio's frames per second (fps) and number of channels
    fps = audio.fps
    n_channels = audio.nchannels
    
    # Assuming 16-bit audio (2 bytes per sample), calculate the bitrate (in bits per second)
    bitrate = fps * n_channels * 2 * 8

    # Calculate the maximum duration (in seconds) for each segment based on the max size (in bytes)
    max_duration_seconds = (max_size_MB * 1024 * 1024 * 8) / bitrate

    # Determine the number of segments
    num_segments = int(audio.duration // max_duration_seconds) + 1

    # Split the audio into segments and save them
    for i in range(num_segments):
        start = i * max_duration_seconds
        end = (i + 1) * max_duration_seconds if (i + 1) * max_duration_seconds < audio.duration else audio.duration
        segment = audio.subclip(start, end)
        segment.write_audiofile(f"output_segment_{i}.{audio_format}", codec=audio_format)

    print(f"{num_segments} audio segments have been created.")

# Example usage
splice_audio("ThriveCapital.mp4", max_size_MB=25)
