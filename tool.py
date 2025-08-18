import os
import tempfile
import json
from typing import Dict, List, Any, Callable
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
import yt_dlp
import librosa
import soundfile as sf
import cv2
import numpy as np
# Correctly import VideoClip alongside the others
from moviepy.editor import VideoClip, VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip
import lyricsgenius

class KaraokeGenerator:
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model="gpt-4",
            temperature=0.3
        )
        # You'll need to get a Genius API key for lyrics
        # self.genius = lyricsgenius.Genius("YOUR_GENIUS_API_KEY")
    
    def get_clarifying_questions(self, song_name: str) -> List[str]:
        """Generate 2-3 clarifying questions about the song."""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a music expert helping to identify the exact song for karaoke generation.
            Given a song name, generate 2-3 specific questions to clarify which exact version/recording the user wants.
            Focus on: artist/performer, specific version, language, or notable covers.
            Keep questions simple and practical."""),
            HumanMessage(content=f"Song name: {song_name}")
        ])
        response = self.llm.invoke(prompt.format_messages())
        
        questions = []
        lines = response.content.strip().split('\n')
        for line in lines:
            if line.strip() and ('?' in line):
                question = line.strip().lstrip('123.-* ')
                questions.append(question)
        return questions[:3]

    def get_song_info(self, song_name: str, answers: Dict[str, str]) -> Dict[str, Any]:
        """Get detailed song information using LLM."""
        answers_text = "\n".join([f"- {answer}" for answer in answers.values() if answer])
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a music database expert. Based on the song name and additional details,
            provide a JSON object with: title, artist, album, year, genre, duration (mm:ss), language, and a popularity score (1-10).
            Return ONLY the JSON object."""),
            HumanMessage(content=f"Song: {song_name}\nAdditional details:\n{answers_text}")
        ])
        response = self.llm.invoke(prompt.format_messages())
        
        try:
            return json.loads(response.content)
        except:
            return {
                "title": song_name, "artist": "Unknown", "album": "Unknown", "year": "Unknown",
                "genre": "Unknown", "duration": "3:30", "language": "English", "popularity": 5
            }

    def get_lyrics(self, song_info: Dict[str, Any]) -> str:
        """Get song lyrics using multiple methods."""
        try:
            if hasattr(self, 'genius'):
                song = self.genius.search_song(song_info['title'], song_info['artist'])
                if song:
                    return song.lyrics
        except:
            pass
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a lyrics database. Provide the complete lyrics for the requested song with proper verse/chorus structure."),
            HumanMessage(content=f"Song: {song_info['title']} by {song_info['artist']}")
        ])
        response = self.llm.invoke(prompt.format_messages())
        return response.content

    def download_audio(self, song_info: Dict[str, Any]) -> str:
        """Download audio using yt-dlp."""
        search_query = f"{song_info['title']} {song_info['artist']} official audio"
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': tempfile.mktemp(suffix='.%(ext)s'),
            'noplaylist': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                search_results = ydl.extract_info(f"ytsearch1:{search_query}", download=False)
                if search_results['entries']:
                    video_url = search_results['entries'][0]['webpage_url']
                    info = ydl.extract_info(video_url, download=True)
                    return ydl.prepare_filename(info)
                else:
                    raise Exception("No search results found for the song on YouTube.")
        except Exception as e:
            raise Exception(f"Failed to download audio: {str(e)}")

    def separate_vocals(self, audio_path: str) -> str:
        """
        Separate vocals from music using librosa's Harmonic-Percussive Source Separation.
        """
        output_dir = tempfile.mkdtemp()
        instrumental_path = os.path.join(output_dir, 'instrumental.wav')
        
        try:
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 10240: # 10 KB threshold
                raise ValueError("Downloaded audio file is missing, empty, or too small.")

            y, sr = librosa.load(audio_path, sr=22050, mono=True)
            y_harmonic, _ = librosa.effects.hpss(y)
            
            sf.write(instrumental_path, y_harmonic, sr)
            
            return instrumental_path
        except Exception as e:
            print(f"A detailed error occurred in librosa: {e}")
            raise Exception(f"Could not process the audio file with librosa. Ensure FFmpeg is installed and accessible in your system's PATH.")

    def create_karaoke_video(self, instrumental_path: str, lyrics: str, song_info: Dict[str, Any]) -> str:
        """Create karaoke video with lyrics and instrumental."""
        audio_clip = AudioFileClip(instrumental_path)
        duration = audio_clip.duration
        background = self.create_background_video(duration)
        timed_lyrics = self.time_lyrics(lyrics, duration)
        
        text_clips = []
        for start_time, end_time, text in timed_lyrics:
            text_clip = TextClip(
                text, fontsize=50, color='white', font='Arial-Bold',
                stroke_color='black', stroke_width=2
            ).set_position('center').set_start(start_time).set_end(end_time)
            text_clips.append(text_clip)
        
        final_video = CompositeVideoClip([background] + text_clips).set_audio(audio_clip)
        
        output_path = tempfile.mktemp(suffix='.mp4')
        final_video.write_videofile(
            output_path, codec='libx264', audio_codec='aac',
            temp_audiofile=tempfile.mktemp(suffix='.m4a'), remove_temp=True,
            verbose=False, logger=None
        )
        return output_path

    def create_background_video(self, duration: float) -> VideoClip:
        """Create a simple, animated gradient background video."""
        width, height = 1920, 1080
        def make_frame(t):
            gradient = np.zeros((height, width, 3), dtype=np.uint8)
            hue = (t * 20) % 360
            color1_hsv = np.array([hue, 255, 100])
            color2_hsv = np.array([(hue + 60) % 360, 255, 150])
            for y in range(height):
                ratio = y / height
                color_hsv = color1_hsv * (1 - ratio) + color2_hsv * ratio
                gradient[y, :] = cv2.cvtColor(np.uint8([[color_hsv]]), cv2.COLOR_HSV2RGB)[0, 0]
            return gradient
        
        # Use VideoClip for generated content, not VideoFileClip
        return VideoClip(make_frame, duration=duration)

    def time_lyrics(self, lyrics: str, duration: float) -> List[tuple]:
        """Evenly time lyrics to fit the song duration."""
        lines = [line.strip() for line in lyrics.split('\n') if line.strip() and not line.startswith('[')]
        if not lines:
            return [(0, duration, "♪ Instrumental ♪")]
        
        time_per_line = duration / len(lines)
        timed_lyrics = []
        for i, line in enumerate(lines):
            start_time = i * time_per_line
            end_time = (i + 1) * time_per_line
            timed_lyrics.append((start_time, end_time, line))
        
        return timed_lyrics

    def generate_karaoke(self, song_name: str, answers: Dict[str, str], 
                        progress_callback: Callable = None) -> Dict[str, Any]:
        """Main method to generate complete karaoke."""
        try:
            if progress_callback: progress_callback("Getting song information...", 0.1)
            song_info = self.get_song_info(song_name, answers)
            
            if progress_callback: progress_callback("Fetching lyrics...", 0.2)
            lyrics = self.get_lyrics(song_info)
            
            if progress_callback: progress_callback("Downloading audio...", 0.4)
            audio_path = self.download_audio(song_info)
            
            if progress_callback: progress_callback("Creating instrumental track...", 0.6)
            instrumental_path = self.separate_vocals(audio_path)
            
            if progress_callback: progress_callback("Generating karaoke video...", 0.8)
            video_path = self.create_karaoke_video(instrumental_path, lyrics, song_info)
            
            if progress_callback: progress_callback("Complete!", 1.0)
            
            return {
                'song_info': song_info, 'lyrics': lyrics,
                'instrumental_path': instrumental_path, 'video_path': video_path,
                'success': True
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}