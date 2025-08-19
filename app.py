import streamlit as st
import os
from tool import KaraokeGenerator
import time
from dotenv import load_dotenv

load_dotenv()

def main():
    st.set_page_config(
        page_title="AI Karaoke Generator",
        page_icon="🎤",
        layout="wide"
    )
    
    st.title("🎤 AI Karaoke Generator")
    st.markdown("Generate high-quality instrumental tracks with synchronized multilingual lyrics display!")
    
    with st.sidebar:
        st.header("✨ What's New")
        st.markdown("""
        - 🎵 **Real Lyrics** via Genius API
        - 🔑 **Secure API Key** handling
        - 🌏 **Multilingual lyrics support**
        - 🎧 **Better audio quality**
        - ⏱️ **LLM-powered timing sync**
        """)
        
        st.header("ℹ️ How it works")
        st.markdown("""
        **Step 1:** Enter a song name
        **Step 2:** Answer clarifying questions
        **Step 3:** Get:
        - 🎵 High-quality instrumental
        - 📝 Accurate, synchronized lyrics
        - 🎤 Real-time karaoke display
        """)

    if not os.getenv("OPENAI_API_KEY") or not os.getenv("GENIUS_ACCESS_TOKEN"):
        st.error("🚨 Missing API Keys!")
        st.info("Please create a `.env` file with your `OPENAI_API_KEY` and `GENIUS_ACCESS_TOKEN` to continue.")
        st.code("OPENAI_API_KEY=\"...\"\nGENIUS_ACCESS_TOKEN=\"...\"")
        return
    
    try:
        karaoke_gen = KaraokeGenerator()
        st.header("🎵 Song Request")
        if 'stage' not in st.session_state:
            st.session_state.stage = 'input'
            st.session_state.song_info = {}
        
        if st.session_state.stage == 'input':
            col1, col2 = st.columns([3, 1])
            
            with col1:
                song_name = st.text_input(
                    "🎶 Enter the song name:",
                    placeholder="e.g., Shape of You, Tum Hi Ho, Despacito...",
                    label_visibility="visible"
                )
            
            with col2:
                st.write("")
                st.write("")
                search_button = st.button("🔍 Search Song", type="primary", use_container_width=True)
            
            st.markdown("**Popular Examples:**")
            example_cols = st.columns(4)
            examples = ["Blinding Lights", "Shape of You", "Despacito", "Instant Crush"]
            
            for i, example in enumerate(examples):
                with example_cols[i]:
                    if st.button(example, key=f"example_{i}", use_container_width=True):
                        st.session_state.example_song = example
                        st.rerun()
            
            if hasattr(st.session_state, 'example_song'):
                song_name = st.session_state.example_song
                del st.session_state.example_song
                search_button = True
            
            if song_name and search_button:
                with st.spinner("🔍 Searching for song information..."):
                    questions = karaoke_gen.get_clarifying_questions(song_name)
                    st.session_state.questions = questions
                    st.session_state.song_name = song_name
                    st.session_state.stage = 'clarify'
                    st.rerun()
        
        elif st.session_state.stage == 'clarify':
            st.info(f"🎵 Song: **{st.session_state.song_name}**")
            st.subheader("📋 Please provide additional details:")
            answers = {}
            for i, question in enumerate(st.session_state.questions):
                answers[f"q{i}"] = st.text_input(question, key=f"question_{i}")
            
            col_back, col_generate = st.columns([1, 2])
            with col_back:
                if st.button("⬅️ Back to Search"):
                    st.session_state.stage = 'input'
                    st.rerun()
            
            with col_generate:
                if st.button("🎤 Generate Karaoke", type="primary"):
                    if all(answers.values()):
                        st.session_state.answers = answers
                        st.session_state.stage = 'generate'
                        st.rerun()
                    else:
                        st.error("Please answer all questions to proceed.")
        
        elif st.session_state.stage == 'generate':
            st.info(f"🎵 Generating karaoke for: **{st.session_state.song_name}**")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def progress_callback(step, progress):
                progress_bar.progress(progress)
                status_text.text(f"🔄 {step}")
            
            try:
                result = karaoke_gen.generate_karaoke(
                    st.session_state.song_name,
                    st.session_state.answers,
                    progress_callback=progress_callback
                )
                st.session_state.result = result
                st.session_state.stage = 'result'
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error generating karaoke: {str(e)}")
                if st.button("🔄 Try Again"):
                    st.session_state.stage = 'clarify'
                    st.rerun()
        
        elif st.session_state.stage == 'result':
            result = st.session_state.result
            
            if result.get('success', False):
                st.success("✅ Karaoke generated successfully!")
                with st.container(border=True):
                    st.markdown("### 🎵 Song Information")
                    info_cols = st.columns(2)
                    with info_cols[0]:
                        st.markdown(f"**🎤 Title:** {result['song_info']['title']}")
                        st.markdown(f"**👨‍🎤 Artist:** {result['song_info']['artist']}")
                    with info_cols[1]:
                        st.markdown(f"**⏰ Duration:** {result['song_info'].get('duration', 'N/A')}")
                        st.markdown(f"**🌏 Language:** {result['song_info'].get('language', 'N/A')}")
                
                st.subheader("🎵 Instrumental Track")
                if result.get('instrumental_path'):
                    with open(result['instrumental_path'], 'rb') as audio_file:
                        audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format='audio/wav')
                    st.download_button(
                        label="⬇️ Download Instrumental",
                        data=audio_bytes,
                        file_name=f"{st.session_state.song_name}_instrumental.wav",
                        mime="audio/wav"
                    )
                
                st.divider()
                st.subheader("🎤 Synchronized Lyrics")
                if 'lyrics_timing' not in st.session_state:
                    st.session_state.lyrics_timing = result.get('timed_lyrics', [])
                
                control_cols = st.columns(4)
                with control_cols[0]:
                    if st.button("▶️ Start Sync", use_container_width=True):
                        st.session_state.lyrics_started = True
                        st.session_state.start_time = time.time()
                with control_cols[1]:
                    if st.button("⏸️ Pause", use_container_width=True):
                        st.session_state.lyrics_started = False
                with control_cols[2]:
                    if st.button("🔄 Reset", use_container_width=True):
                        st.session_state.lyrics_started = False
                        if 'start_time' in st.session_state: del st.session_state.start_time
                
                lyrics_container = st.container(border=True)
                current_time = (time.time() - st.session_state.start_time) if st.session_state.get('lyrics_started') else 0

                with lyrics_container:
                    current_lyric, next_lyric = "", ""
                    if st.session_state.lyrics_timing:
                        for i, (start, end, text) in enumerate(st.session_state.lyrics_timing):
                            if start <= current_time < end:
                                current_lyric = text
                                if i + 1 < len(st.session_state.lyrics_timing):
                                    next_lyric = st.session_state.lyrics_timing[i + 1][2]
                                break
                    
                    lyrics_html = f"""
                    <div style="text-align: center; padding: 20px; border-radius: 10px; margin: 10px 0;">
                        <div style="font-size: 28px; font-weight: bold; margin-bottom: 10px;">
                            🎤 {current_lyric if current_lyric else "♪ ♫ ♪"}
                        </div>
                        <div style="font-size: 18px; opacity: 0.7;">
                            {f'Next: {next_lyric}' if next_lyric else ''}
                        </div>
                    </div>
                    """
                    st.markdown(lyrics_html, unsafe_allow_html=True)

                if st.session_state.get('lyrics_started'):
                    time.sleep(0.1)
                    st.rerun()
                
                with st.expander("View Full Lyrics", expanded=False):
                    st.text_area("Lyrics:", value=result['lyrics'], height=300)
                
                st.divider()
                if st.button("🎵 Generate Another Song"):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()
            else:
                st.error(f"❌ Failed to generate karaoke: {result.get('error', 'Unknown error')}")
                if st.button("🔄 Try Again"):
                    st.session_state.stage = 'clarify'
                    st.rerun()
    
    except Exception as e:
        st.error(f"❌ An application error occurred: {str(e)}")
        st.info("Please refresh the page and try again.")
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        🎤 <strong>AI Karaoke Generator</strong>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()