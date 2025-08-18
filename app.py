import streamlit as st
import os
from tool import KaraokeGenerator
from langchain.chat_models import ChatOpenAI
import tempfile
import base64

def main():
    st.set_page_config(
        page_title="🎤 AI Karaoke Generator",
        page_icon="🎤",
        layout="wide"
    )
    
    st.title("🎤 AI Karaoke Generator")
    st.markdown("Generate professional karaoke videos with instrumental tracks and synchronized lyrics!")
    
    # Sidebar for API key
    with st.sidebar:
        st.header("🔑 Configuration")
        openai_api_key = st.text_input(
            "Enter your OpenAI API Key:",
            type="password",
            help="Get your API key from https://platform.openai.com/account/api-keys"
        )
        
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            st.success("✅ API Key configured!")
        else:
            st.warning("⚠️ Please enter your OpenAI API Key to continue")
    
    if not openai_api_key:
        st.info("👈 Please enter your OpenAI API Key in the sidebar to get started")
        return
    
    # Initialize the karaoke generator
    try:
        karaoke_gen = KaraokeGenerator(openai_api_key)
        
        # Main interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("🎵 Song Request")
            
            # Initialize session state
            if 'stage' not in st.session_state:
                st.session_state.stage = 'input'
                st.session_state.song_info = {}
            
            # Stage 1: Initial song input
            if st.session_state.stage == 'input':
                song_name = st.text_input(
                    "🎶 Enter the song name:",
                    placeholder="e.g., Shape of You, Bohemian Rhapsody, Despacito..."
                )
                
                if song_name and st.button("🔍 Search Song", type="primary"):
                    with st.spinner("Searching for song information..."):
                        # Get clarifying questions
                        questions = karaoke_gen.get_clarifying_questions(song_name)
                        st.session_state.questions = questions
                        st.session_state.song_name = song_name
                        st.session_state.stage = 'clarify'
                        st.rerun()
            
            # Stage 2: Clarifying questions
            elif st.session_state.stage == 'clarify':
                st.info(f"🎵 Song: **{st.session_state.song_name}**")
                st.subheader("📋 Please provide additional details:")
                
                answers = {}
                for i, question in enumerate(st.session_state.questions):
                    answers[f"q{i}"] = st.text_input(question, key=f"question_{i}")
                
                col_back, col_generate = st.columns([1, 2])
                
                with col_back:
                    if st.button("⬅️ Back"):
                        st.session_state.stage = 'input'
                        st.rerun()
                
                with col_generate:
                    if st.button("🎤 Generate Karaoke", type="primary"):
                        # Check if all questions are answered
                        if all(answers.values()):
                            st.session_state.answers = answers
                            st.session_state.stage = 'generate'
                            st.rerun()
                        else:
                            st.error("Please answer all questions before proceeding.")
            
            # Stage 3: Generate karaoke
            elif st.session_state.stage == 'generate':
                st.info(f"🎵 Generating karaoke for: **{st.session_state.song_name}**")
                
                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Generate karaoke
                    result = karaoke_gen.generate_karaoke(
                        st.session_state.song_name,
                        st.session_state.answers,
                        progress_callback=lambda step, progress: (
                            progress_bar.progress(progress),
                            status_text.text(step)
                        )
                    )
                    
                    st.session_state.result = result
                    st.session_state.stage = 'result'
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error generating karaoke: {str(e)}")
                    if st.button("🔄 Try Again"):
                        st.session_state.stage = 'clarify'
                        st.rerun()
            
            # Stage 4: Show results
            elif st.session_state.stage == 'result':
                result = st.session_state.result
                
                if result.get('success', False):
                    st.success("✅ Karaoke generated successfully!")
                    
                    # Display song info
                    st.subheader("📋 Song Information")
                    info_col1, info_col2 = st.columns(2)
                    
                    with info_col1:
                        st.write(f"**Title:** {result['song_info']['title']}")
                        st.write(f"**Artist:** {result['song_info']['artist']}")
                    
                    with info_col2:
                        st.write(f"**Genre:** {result['song_info']['genre']}")
                        st.write(f"**Duration:** {result['song_info']['duration']}")
                else:
                    st.error(f"❌ Error generating karaoke: {result.get('error', 'Unknown error')}")
                    if st.button("🔄 Try Again"):
                        st.session_state.stage = 'clarify'
                        st.rerun()
                    return
                
                # Display lyrics
                st.subheader("📝 Lyrics")
                st.text_area("", result['lyrics'], height=300, disabled=True)
                
                # Audio player for instrumental
                st.subheader("🎵 Instrumental Track")
                if result.get('instrumental_path'):
                    with open(result['instrumental_path'], 'rb') as audio_file:
                        audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format='audio/mp3')
                
                # Video player for karaoke
                st.subheader("🎤 Karaoke Video")
                if result.get('video_path'):
                    with open(result['video_path'], 'rb') as video_file:
                        video_bytes = video_file.read()
                        st.video(video_bytes)
                    
                    # Download button
                    st.download_button(
                        label="⬇️ Download Karaoke Video",
                        data=video_bytes,
                        file_name=f"{st.session_state.song_name}_karaoke.mp4",
                        mime="video/mp4"
                    )
                
                # New song button
                if st.button("🎵 Generate Another Song"):
                    st.session_state.stage = 'input'
                    st.rerun()
        
        with col2:
            st.header("ℹ️ How it works")
            st.markdown("""
            **Step 1:** Enter a song name
            
            **Step 2:** Answer clarifying questions about the artist, version, etc.
            
            **Step 3:** AI generates:
            - 🎵 Instrumental track
            - 📝 Synchronized lyrics
            - 🎥 Karaoke video
            
            **Features:**
            - High-quality instrumental separation
            - Synchronized lyric display
            - Professional video output
            - Download capability
            """)
            
            st.header("💡 Tips")
            st.markdown("""
            - Be specific with song titles
            - Include artist name if asked
            - Popular songs work best
            - Check spelling carefully
            """)
    
    except Exception as e:
        st.error(f"❌ Error initializing application: {str(e)}")

if __name__ == "__main__":
    main()