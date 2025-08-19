# 🎤 AI Karaoke Generator
An AI-powered application that generates professional karaoke videos with instrumental tracks and synchronized lyrics.

## Features

- 🎵 **Smart Song Recognition**: AI helps identify the exact song you want
- 🎼 **Instrumental Generation**: Separates vocals from music using advanced audio processing
- 📝 **Lyric Synchronization**: Automatically times lyrics to match the song
- 🎥 **Video Generation**: Creates professional karaoke videos
- ⬇️ **Download Support**: Save your karaoke videos locally

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Get required API keys:
   - OpenAI API key from https://platform.openai.com/account/api-keys
   - (Optional) Genius API key for better lyrics fetching

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Enter your OpenAI API key** in the sidebar
2. **Input a song name** you want to convert to karaoke
3. **Answer clarifying questions** about the artist, version, etc.
4. **Wait for generation** - the AI will:
   - Find and download the song
   - Separate vocals from instruments
   - Fetch or generate lyrics
   - Create a synchronized karaoke video
5. **Download your karaoke** and start singing!

## Tech Stack

- **Frontend**: Streamlit
- **AI**: LangChain + OpenAI GPT-4
- **Audio Processing**: Spleeter, Librosa
- **Video Generation**: MoviePy, OpenCV
- **Audio Download**: yt-dlp

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key
- `GENIUS_API_KEY`: (Optional) Genius API key for lyrics

### Supported Formats
- **Audio Input**: MP3, WAV, M4A
- **Video Output**: MP4
- **Languages**: Multi-language support

## How It Works

1. **Context Engineering**: The AI asks clarifying questions to identify the exact song
2. **Audio Processing**: Downloads high-quality audio and separates vocals using Spleeter
3. **Lyric Processing**: Fetches lyrics and times them to match the song duration
4. **Video Generation**: Creates a karaoke video with synchronized lyrics overlay

## Limitations

- Requires internet connection for song downloads
- Quality depends on source audio availability
- Some songs may not separate perfectly
- Processing time varies based on song length

