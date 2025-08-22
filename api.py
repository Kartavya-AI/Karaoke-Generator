import os
import tempfile
import json
import asyncio
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager
import logging
import traceback
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, validator
import structlog

from tool import KaraokeGenerator

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

karaoke_generator: Optional[KaraokeGenerator] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global karaoke_generator
    logger.info("Starting Karaoke API server...")
    
    try:
        karaoke_generator = KaraokeGenerator()
        logger.info("KaraokeGenerator initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize KaraokeGenerator", error=str(e))
        raise RuntimeError(f"Failed to initialize application: {str(e)}")
    
    yield
    
    logger.info("Shutting down Karaoke API server...")
    cleanup_temp_files()

def cleanup_temp_files():
    try:
        temp_dir = tempfile.gettempdir()
        temp_path = Path(temp_dir)
        for file in temp_path.glob("tmp*"):
            if file.suffix in ['.wav', '.mp3', '.m4a']:
                try:
                    file.unlink()
                    logger.info(f"Cleaned up temporary file: {file}")
                except Exception as e:
                    logger.warning(f"Failed to clean up {file}: {e}")
                    
    except Exception as e:
        logger.error("Error during cleanup", error=str(e))

app = FastAPI(
    title="Karaoke Generator API",
    description="Professional karaoke track generation API with vocal separation and synchronized lyrics",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] 
)

class SongRequest(BaseModel):
    song_name: str = Field(..., min_length=1, max_length=200, description="Name of the song")
    
    @validator('song_name')
    def validate_song_name(cls, v):
        if not v.strip():
            raise ValueError('Song name cannot be empty or just whitespace')
        return v.strip()

class ClarifyingQuestionsResponse(BaseModel):
    questions: List[str] = Field(..., description="List of clarifying questions")
    song_name: str = Field(..., description="Original song name")

class AnswersRequest(BaseModel):
    song_name: str = Field(..., min_length=1, max_length=200)
    answers: Dict[str, str] = Field(..., description="Answers to clarifying questions")
    
    @validator('song_name')
    def validate_song_name(cls, v):
        return v.strip()

class SongInfo(BaseModel):
    title: str
    artist: str
    album: str
    year: str
    genre: str
    duration: str
    language: str
    popularity: int

class TimedLyric(BaseModel):
    start: float
    end: float
    text: str

class KaraokeResponse(BaseModel):
    success: bool
    song_info: Optional[SongInfo] = None
    lyrics: Optional[str] = None
    timed_lyrics: Optional[List[TimedLyric]] = None
    instrumental_download_url: Optional[str] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: str

generated_files: Dict[str, str] = {}

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    logger.error("ValueError occurred", error=str(exc), path=str(request.url))
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "Invalid input", "detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error("Unexpected error occurred", 
                error=str(exc), 
                traceback=traceback.format_exc(),
                path=str(request.url))
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"}
    )

@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "message": "Karaoke Generator API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        environment=os.getenv("ENVIRONMENT", "production")
    )

@app.post("/clarifying-questions", response_model=ClarifyingQuestionsResponse)
async def get_clarifying_questions(request: SongRequest):
    global karaoke_generator
    
    if not karaoke_generator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )
    
    try:
        logger.info("Getting clarifying questions", song_name=request.song_name)
        questions = await asyncio.get_event_loop().run_in_executor(
            None, 
            karaoke_generator.get_clarifying_questions, 
            request.song_name
        )
        
        logger.info("Successfully generated clarifying questions", 
                   song_name=request.song_name, 
                   num_questions=len(questions))
        
        return ClarifyingQuestionsResponse(
            questions=questions,
            song_name=request.song_name
        )
        
    except Exception as e:
        logger.error("Error getting clarifying questions", 
                    song_name=request.song_name, 
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate clarifying questions: {str(e)}"
        )

@app.post("/generate-karaoke", response_model=KaraokeResponse)
async def generate_karaoke(request: AnswersRequest, background_tasks: BackgroundTasks):
    global karaoke_generator
    
    if not karaoke_generator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )
    
    try:
        logger.info("Starting karaoke generation", 
                   song_name=request.song_name,
                   answers=request.answers)
        progress_info = {"status": "starting", "progress": 0.0}
        
        def progress_callback(message: str, progress: float):
            progress_info["status"] = message
            progress_info["progress"] = progress
            logger.info("Generation progress", message=message, progress=progress)
        
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            karaoke_generator.generate_karaoke,
            request.song_name,
            request.answers,
            progress_callback
        )
        
        if not result.get('success', False):
            error_msg = result.get('error', 'Unknown error occurred')
            logger.error("Karaoke generation failed", 
                        song_name=request.song_name, 
                        error=error_msg)
            return KaraokeResponse(success=False, error=error_msg)
        
        file_key = f"instrumental_{hash(request.song_name + str(request.answers))}"
        generated_files[file_key] = result['instrumental_path']
        
        background_tasks.add_task(cleanup_file_after_delay, file_key, 3600)
        timed_lyrics = [
            TimedLyric(start=start, end=end, text=text)
            for start, end, text in result['timed_lyrics']
        ]
        
        song_info = SongInfo(**result['song_info'])
        
        logger.info("Karaoke generation completed successfully", 
                   song_name=request.song_name,
                   file_key=file_key)
        
        return KaraokeResponse(
            success=True,
            song_info=song_info,
            lyrics=result['lyrics'],
            timed_lyrics=timed_lyrics,
            instrumental_download_url=f"/download/{file_key}"
        )
        
    except Exception as e:
        logger.error("Error during karaoke generation", 
                    song_name=request.song_name, 
                    error=str(e),
                    traceback=traceback.format_exc())
        return KaraokeResponse(
            success=False, 
            error=f"Failed to generate karaoke: {str(e)}"
        )

@app.get("/download/{file_key}")
async def download_instrumental(file_key: str):
    if file_key not in generated_files:
        logger.warning("Requested file not found", file_key=file_key)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found or has expired"
        )
    
    file_path = generated_files[file_key]
    
    if not os.path.exists(file_path):
        logger.warning("Physical file not found", file_key=file_key, file_path=file_path)
        del generated_files[file_key]
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found"
        )
    
    logger.info("Serving instrumental file", file_key=file_key)
    
    return FileResponse(
        path=file_path,
        media_type='audio/wav',
        filename=f"instrumental_{file_key}.wav",
        headers={"Cache-Control": "no-cache"}
    )

@app.delete("/cleanup/{file_key}")
async def cleanup_file(file_key: str):
    
    if file_key not in generated_files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found"
        )
    
    file_path = generated_files[file_key]
    
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        del generated_files[file_key]
        logger.info("File cleaned up successfully", file_key=file_key)
        return {"message": "File cleaned up successfully"}
    except Exception as e:
        logger.error("Error cleaning up file", file_key=file_key, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup file: {str(e)}"
        )

async def cleanup_file_after_delay(file_key: str, delay_seconds: int):
    await asyncio.sleep(delay_seconds)    
    if file_key in generated_files:
        try:
            file_path = generated_files[file_key]
            if os.path.exists(file_path):
                os.remove(file_path)
            del generated_files[file_key]
            logger.info("Scheduled file cleanup completed", file_key=file_key)
        except Exception as e:
            logger.error("Error in scheduled cleanup", file_key=file_key, error=str(e))

@app.get("/files/status")
async def get_files_status():
    active_files = []
    
    for file_key, file_path in generated_files.items():
        exists = os.path.exists(file_path)
        size = os.path.getsize(file_path) if exists else 0
        active_files.append({
            "key": file_key,
            "exists": exists,
            "size_bytes": size,
            "path": file_path
        })
    
    return {
        "active_files_count": len(generated_files),
        "files": active_files
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )