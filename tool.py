import os
import tempfile
import json
import re
from typing import Dict, List, Any, Callable, Tuple
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
import yt_dlp
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, sosfilt
import lyricsgenius
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class KaraokeGenerator:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.genius_access_token = os.getenv("GENIUS_ACCESS_TOKEN")
        if not self.openai_api_key or not self.genius_access_token:
            raise ValueError("OPENAI_API_KEY and GENIUS_ACCESS_TOKEN must be set in your environment.")
        
        self.llm = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            model="gpt-4o-mini",
            temperature=0.3
        )
        self.genius = lyricsgenius.Genius(self.genius_access_token, verbose=False, remove_section_headers=True)

    def get_clarifying_questions(self, song_name: str) -> List[str]:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a music expert helping to identify the exact song for karaoke generation.
            Given a song name, generate 2-3 specific questions to clarify which exact version/recording the user wants.
            Focus on: artist/performer, specific version, or language.
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
        answers_text = "\n".join([f"- {answer}" for answer in answers.values() if answer])
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a music database expert. Based on the song name and additional details,
            provide a JSON object with EXACTLY these fields:
            - title: string (song title)
            - artist: string (artist name)
            - album: string (album name, or "Single" if unknown)
            - year: string (year or "Unknown")
            - genre: string (music genre)
            - duration: string (format: "mm:ss", estimate if unknown)
            - language: string (primary language of the song)
            - popularity: number (integer from 1-10)
            
            Return ONLY the JSON object, no additional text. Ensure all fields are included."""),
            HumanMessage(content=f"Song: {song_name}\nAdditional details:\n{answers_text}")
        ])
        response = self.llm.invoke(prompt.format_messages())
        
        default_info = {
            "title": song_name, "artist": "Unknown Artist", "album": "Unknown Album", 
            "year": "Unknown", "genre": "Pop", "duration": "3:30", 
            "language": "English", "popularity": 5
        }
        
        try:
            content = response.content.strip()
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'): content = content[4:]
            content = content.strip()
            parsed_info = json.loads(content)
            
            for key, default_value in default_info.items():
                if key not in parsed_info: parsed_info[key] = default_value
                if key == 'popularity':
                    try: parsed_info[key] = min(10, max(1, int(parsed_info[key])))
                    except: parsed_info[key] = 5
            return parsed_info
        except Exception:
            default_info["title"] = song_name
            return default_info

    def get_lyrics(self, song_info: Dict[str, Any]) -> str:
        try:
            song = self.genius.search_song(song_info['title'], song_info['artist'])
            if song:
                lyrics = re.sub(r'\[.*?\]', '', song.lyrics).strip()
                lines = lyrics.split('\n')
                if len(lines) > 1 and lines[0].strip().lower() == song_info['title'].lower():
                    lyrics = '\n'.join(lines[1:]).strip()
                return lyrics
            else:
                return "Lyrics not found for this song."
        except Exception as e:
            print(f"Error fetching lyrics from Genius: {e}")
            return "Could not fetch lyrics."

    def download_audio(self, song_info: Dict[str, Any]) -> str:
        search_query = f"{song_info.get('title', '')} {song_info.get('artist', '')} official audio"
        temp_path = tempfile.mktemp(suffix='.%(ext)s')
        
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio/best',
            'outtmpl': temp_path, 'noplaylist': True, 'extract_flat': False,
            'prefer_ffmpeg': True,
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '192'}],
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                search_results = ydl.extract_info(f"ytsearch1:{search_query}", download=False)
                if not search_results.get('entries'):
                    raise Exception("No search results found on YouTube.")
                
                video_url = search_results['entries'][0]['webpage_url']
                ydl.extract_info(video_url, download=True)
                wav_path = temp_path.replace('.%(ext)s', '.wav')
                
                if os.path.exists(wav_path): return wav_path
                else: raise Exception("Downloaded audio file not found.")
        except Exception as e:
            raise Exception(f"Failed to download audio: {str(e)}")

    def separate_vocals_advanced(self, audio_path: str) -> str:
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 10240:
            raise ValueError("Downloaded audio file is missing or empty.")
        
        try:
            y, sr = librosa.load(audio_path, sr=44100, mono=False)
            if y.ndim > 1 and y.shape[0] == 2:
                y_instrumental = y[0] - y[1]

            else:
                y_mono = librosa.to_mono(y)
                y_harmonic, y_percussive = librosa.effects.hpss(y_mono, margin=3.0)
                S_harmonic = librosa.stft(y_harmonic)
                S_vocals_approx = np.median(np.abs(S_harmonic), axis=1, keepdims=True) * np.ones_like(S_harmonic)
                mask = np.minimum(1.0, 2.0 * np.abs(S_vocals_approx) / (np.abs(S_harmonic) + 1e-8))
                S_harmonic_instrumental = S_harmonic * (1 - mask)
                y_harmonic_instrumental = librosa.istft(S_harmonic_instrumental)
                y_instrumental = y_percussive + y_harmonic_instrumental

            max_val = np.max(np.abs(y_instrumental))
            if max_val > 0:
                y_instrumental = y_instrumental / max_val * 0.95
            output_path = tempfile.mktemp(suffix='.wav')
            sf.write(output_path, y_instrumental, sr, format='WAV', subtype='PCM_16')
            return output_path
            
        except Exception as e:
            raise Exception(f"Could not process the audio file: {str(e)}")

    def create_synchronized_lyrics(self, lyrics: str, duration_str: str) -> List[Tuple[float, float, str]]:
        try:
            minutes, seconds = map(int, duration_str.split(':'))
            total_duration = minutes * 60 + seconds
        except:
            total_duration = 210
        
        lines = [line for line in lyrics.split('\n') if line.strip() and len(line.strip()) > 2]
        if not lines:
            return [(0, total_duration, "🎵 Instrumental Track 🎵")]

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""You are a karaoke timing expert. Your task is to analyze the provided song lyrics and create a synchronized timeline.
            The total song duration is {total_duration} seconds.
            Analyze the structure (verses, chorus, pauses) to estimate the start and end time for each line.
            - Add an 8-second instrumental intro.
            - Leave a 5-second instrumental outro.
            - A chorus line is typically faster than a verse line.
            - Add small pauses (0.5s) between lines.
            
            Return a JSON array of objects, where each object has "start" (float), "end" (float), and "text" (string) for each line.
            The final "end" time should not exceed {total_duration}.
            
            Example format:
            [
              {{"start": 0.0, "end": 8.0, "text": "🎵 Intro 🎵"}},
              {{"start": 8.5, "end": 12.0, "text": "First line of lyrics"}},
              ...
            ]
            Return ONLY the JSON array."""),
            HumanMessage(content=f"Lyrics:\n{lyrics}")
        ])
        
        try:
            response = self.llm.invoke(prompt.format_messages())
            content = response.content.strip()
            json_start_index = content.find('[')
            json_end_index = content.rfind(']')
            
            if json_start_index != -1 and json_end_index != -1:
                json_string = content[json_start_index : json_end_index + 1]
                timed_data = json.loads(json_string)
                timed_lyrics = [(item['start'], item['end'], item['text']) for item in timed_data]
                return timed_lyrics
            else:
                raise ValueError("LLM response did not contain a valid JSON array.")

        except Exception as e:
            print(f"Error creating synchronized lyrics with LLM: {e}. Falling back to basic timing.")
            timed_lyrics = [(0, 8.0, "🎵 Get ready to sing! 🎵")]
            time_per_line = (total_duration - 16) / len(lines) if lines else 5.0
            current_time = 8.0
            for line in lines:
                end_time = current_time + time_per_line
                timed_lyrics.append((current_time, end_time, line))
                current_time = end_time + 0.5
            timed_lyrics.append((current_time, total_duration, "🎵 Thank you for singing! 🎵"))
            return timed_lyrics

    def generate_karaoke(self, song_name: str, answers: Dict[str, str], progress_callback: Callable = None) -> Dict[str, Any]:
        try:
            if progress_callback: progress_callback("Getting song information...", 0.1)
            song_info = self.get_song_info(song_name, answers)
            
            if progress_callback: progress_callback("Fetching accurate lyrics...", 0.2)
            lyrics = self.get_lyrics(song_info)
            if "not found" in lyrics.lower() or "could not fetch" in lyrics.lower():
                raise Exception("Could not retrieve lyrics for this song.")
            
            if progress_callback: progress_callback("Downloading high-quality audio...", 0.4)
            audio_path = self.download_audio(song_info)
            
            if progress_callback: progress_callback("Creating instrumental track...", 0.7)
            instrumental_path = self.separate_vocals_advanced(audio_path)
            
            if progress_callback: progress_callback("Synchronizing lyrics...", 0.9)
            timed_lyrics = self.create_synchronized_lyrics(lyrics, song_info.get('duration', '3:30'))
            
            if progress_callback: progress_callback("Complete!", 1.0)
            
            if os.path.exists(audio_path): os.remove(audio_path)
            
            return {
                'song_info': song_info, 'lyrics': lyrics, 'timed_lyrics': timed_lyrics,
                'instrumental_path': instrumental_path, 'success': True
            }
        except Exception as e:
            error_msg = f"Error generating karaoke: {str(e)}"
            print(error_msg)
            return {'success': False, 'error': error_msg}