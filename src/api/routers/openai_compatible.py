"""OpenAI-compatible router for text-to-speech"""

import io
import json
import os
import re
import tempfile
from typing import AsyncGenerator, Dict, List, Union

import aiofiles
import torch
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from loguru import logger

from ..core.config import settings
from ..services.audio import AudioService
from ..services.tts_service import TTSService
from ..structures import OpenAISpeechRequest

# Memory monitoring
import psutil
MEMORY_THRESHOLD = 0.9  # 90% memory usage threshold

# Rate limiting
from fastapi.middleware.throttling import ThrottlingMiddleware
from datetime import datetime, timedelta
from collections import defaultdict

# Global rate limiting state
request_counts = defaultdict(lambda: {"count": 0, "reset_time": datetime.now()})
RATE_LIMIT = 100  # requests per minute
RATE_WINDOW = 60  # seconds

# Load OpenAI mappings
def load_openai_mappings() -> Dict:
    """Load OpenAI voice and model mappings from JSON"""
    api_dir = os.path.dirname(os.path.dirname(__file__))
    mapping_path = os.path.join(api_dir, "core", "openai_mappings.json")
    try:
        with open(mapping_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load OpenAI mappings: {e}")
        return {"models": {}, "voices": {}}

# Global mappings
_openai_mappings = load_openai_mappings()

router = APIRouter(
    tags=["OpenAI Compatible TTS"],
    responses={404: {"description": "Not found"}},
)

# Global TTSService instance with lock
_tts_service = None
_init_lock = None

def check_rate_limit(client_ip: str) -> bool:
    """Check if request is within rate limit"""
    now = datetime.now()
    client_state = request_counts[client_ip]
    
    # Reset counter if window expired
    if now - client_state["reset_time"] > timedelta(seconds=RATE_WINDOW):
        client_state["count"] = 0
        client_state["reset_time"] = now
    
    # Check limit
    if client_state["count"] >= RATE_LIMIT:
        return False
    
    client_state["count"] += 1
    return True

def check_memory_usage() -> bool:
    """Check if memory usage is below threshold"""
    memory = psutil.virtual_memory()
    return memory.percent < (MEMORY_THRESHOLD * 100)

async def cleanup_temp_files(background_tasks: BackgroundTasks, temp_path: str):
    """Add temp file cleanup to background tasks"""
    async def _cleanup():
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.debug(f"Cleaned up temp file: {temp_path}")
        except Exception as e:
            logger.error(f"Failed to cleanup temp file {temp_path}: {e}")
    
    background_tasks.add_task(_cleanup)


async def get_tts_service() -> TTSService:
    """Get global TTSService instance"""
    global _tts_service, _init_lock

    # Create lock if needed
    if _init_lock is None:
        import asyncio

        _init_lock = asyncio.Lock()

    # Initialize service if needed
    if _tts_service is None:
        async with _init_lock:
            # Double check pattern
            if _tts_service is None:
                _tts_service = await TTSService.create()
                logger.info("Created global TTSService instance")

    return _tts_service


def get_model_name(model: str) -> str:
    """Get internal model name from OpenAI model name"""
    base_name = _openai_mappings["models"].get(model)
    if not base_name:
        raise ValueError(f"Unsupported model: {model}")
    return base_name + ".pth"


async def process_voices(
    voice_input: Union[str, List[str]], tts_service: TTSService
) -> str:
    """Process voice input, handling both string and list formats

    Returns:
        Voice name to use (with weights if specified)
    """
    # Convert input to list of voices
    if isinstance(voice_input, str):
        # Check if it's an OpenAI voice name
        mapped_voice = _openai_mappings["voices"].get(voice_input)
        if mapped_voice:
            voice_input = mapped_voice
        # Split on + but preserve any parentheses
        voices = []
        for part in voice_input.split("+"):
            part = part.strip()
            if not part:
                continue
            # Extract voice name without weight
            voice_name = part.split("(")[0].strip()
            # Check if it's a valid voice
            available_voices = await tts_service.list_voices()
            if voice_name not in available_voices:
                raise ValueError(
                    f"Voice '{voice_name}' not found. Available voices: {', '.join(sorted(available_voices))}"
                )
            voices.append(part)
    else:
        # For list input, map each voice if it's an OpenAI voice name
        voices = []
        for v in voice_input:
            mapped = _openai_mappings["voices"].get(v, v)
            voice_name = mapped.split("(")[0].strip()
            # Check if it's a valid voice
            available_voices = await tts_service.list_voices()
            if voice_name not in available_voices:
                raise ValueError(
                    f"Voice '{voice_name}' not found. Available voices: {', '.join(sorted(available_voices))}"
                )
            voices.append(mapped)

    if not voices:
        raise ValueError("No voices provided")

    # For multiple voices, combine them with +
    return "+".join(voices)


async def stream_audio_chunks(
    tts_service: TTSService, request: OpenAISpeechRequest, client_request: Request
) -> AsyncGenerator[bytes, None]:
    """Stream audio chunks as they're generated with client disconnect handling"""
    voice_name = await process_voices(request.voice, tts_service)

    try:
        logger.info(f"Starting audio generation with lang_code: {request.lang_code}")
        async for chunk in tts_service.generate_audio_stream(
            text=request.input,
            voice=voice_name,
            speed=request.speed,
            output_format=request.response_format,
            lang_code=request.lang_code or settings.default_voice_code or voice_name[0].lower(),
            normalization_options=request.normalization_options
        ):
            # Check if client is still connected
            is_disconnected = client_request.is_disconnected
            if callable(is_disconnected):
                is_disconnected = await is_disconnected()
            if is_disconnected:
                logger.info("Client disconnected, stopping audio generation")
                break
            yield chunk
    except Exception as e:
        logger.error(f"Error in audio streaming: {str(e)}")
        # Let the exception propagate to trigger cleanup
        raise


@router.post("/audio/speech")
async def create_speech(
    request: OpenAISpeechRequest,
    client_request: Request,
    background_tasks: BackgroundTasks,
    x_raw_response: str = Header(None, alias="x-raw-response"),
):
    """OpenAI-compatible endpoint for text-to-speech"""
    # Check rate limit
    client_ip = client_request.client.host
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limit_exceeded",
                "message": f"Rate limit exceeded. Maximum {RATE_LIMIT} requests per {RATE_WINDOW} seconds.",
                "type": "rate_limit_error",
            },
        )
    
    # Check memory usage
    if not check_memory_usage():
        raise HTTPException(
            status_code=503,
            detail={
                "error": "resource_exhausted",
                "message": "Server is currently experiencing high memory usage. Please try again later.",
                "type": "server_error",
            },
        )

    # Validate model before processing request
    if request.model not in _openai_mappings["models"]:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_model",
                "message": f"Unsupported model: {request.model}",
                "type": "invalid_request_error",
            },
        )

    try:
        tts_service = await get_tts_service()
        voice_name = await process_voices(request.voice, tts_service)

        # Set content type based on format
        content_type = {
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav",
            "pcm": "audio/pcm",
        }.get(request.response_format, f"audio/{request.response_format}")

        # Use /tmp directory for Railway compatibility
        temp_dir = "/tmp/kokoro_audio"
        os.makedirs(temp_dir, exist_ok=True)

        if request.stream:
            generator = stream_audio_chunks(tts_service, request, client_request)

            if request.return_download_link:
                from ..services.temp_manager import TempFileWriter

                output_format = request.download_format or request.response_format
                temp_writer = TempFileWriter(output_format, base_dir=temp_dir)
                await temp_writer.__aenter__()

                download_path = temp_writer.download_path

                # Add cleanup task
                background_tasks.add_task(cleanup_temp_files, background_tasks, temp_writer.file_path)

                headers = {
                    "Content-Disposition": f"attachment; filename=speech.{output_format}",
                    "X-Accel-Buffering": "no",
                    "Cache-Control": "no-cache",
                    "Transfer-Encoding": "chunked",
                    "X-Download-Path": download_path,
                }

                async def dual_output():
                    try:
                        async for chunk in generator:
                            if chunk:
                                await temp_writer.write(chunk)
                                yield chunk
                        await temp_writer.finalize()
                    except Exception as e:
                        logger.error(f"Error in dual output streaming: {e}")
                        await temp_writer.__aexit__(type(e), e, e.__traceback__)
                        raise
                    finally:
                        if not temp_writer._finalized:
                            await temp_writer.__aexit__(None, None, None)

                return StreamingResponse(
                    dual_output(), media_type=content_type, headers=headers
                )

            return StreamingResponse(
                generator,
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                    "X-Accel-Buffering": "no",
                    "Cache-Control": "no-cache",
                    "Transfer-Encoding": "chunked",
                },
            )
        else:
            # Generate complete audio using public interface
            audio, _ = await tts_service.generate_audio(
                text=request.input,
                voice=voice_name,
                speed=request.speed,
                lang_code=request.lang_code,
            )

            # Convert to requested format with proper finalization
            content = await AudioService.convert_audio(
                audio,
                24000,
                request.response_format,
                is_first_chunk=True,
                is_last_chunk=True,
            )

            return Response(
                content=content,
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                    "Cache-Control": "no-cache",  # Prevent caching
                },
            )

    except ValueError as e:
        # Handle validation errors
        logger.warning(f"Invalid request: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_error",
                "message": str(e),
                "type": "invalid_request_error",
            },
        )
    except RuntimeError as e:
        # Handle runtime/processing errors
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": str(e),
                "type": "server_error",
            },
        )
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error in speech generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": str(e),
                "type": "server_error",
            },
        )


@router.get("/download/{filename}")
async def download_audio_file(filename: str):
    """Download a generated audio file from temp storage"""
    try:
        from ..core.paths import _find_file, get_content_type

        # Search for file in temp directory
        file_path = await _find_file(
            filename=filename, search_paths=[settings.temp_file_dir]
        )

        # Get content type from path helper
        content_type = await get_content_type(file_path)

        return FileResponse(
            file_path,
            media_type=content_type,
            filename=filename,
            headers={
                "Cache-Control": "no-cache",
                "Content-Disposition": f"attachment; filename={filename}",
            },
        )

    except Exception as e:
        logger.error(f"Error serving download file {filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "server_error",
                "message": "Failed to serve audio file",
                "type": "server_error",
            },
        )


@router.get("/models")
async def list_models():
    """List all available models"""
    try:
        # Create standard model list
        models = [
            {
                "id": "tts-1",
                "object": "model",
                "created": 1686935002,
                "owned_by": "kokoro"
            },
            {
                "id": "tts-1-hd",
                "object": "model",
                "created": 1686935002,
                "owned_by": "kokoro"
            },
            {
                "id": "kokoro",
                "object": "model",
                "created": 1686935002,
                "owned_by": "kokoro"
            }
        ]
        
        return {
            "object": "list",
            "data": models
        }
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "server_error",
                "message": "Failed to retrieve model list",
                "type": "server_error",
            },
        )

@router.get("/models/{model}")
async def retrieve_model(model: str):
    """Retrieve a specific model"""
    try:
        # Define available models
        models = {
            "tts-1": {
                "id": "tts-1",
                "object": "model",
                "created": 1686935002,
                "owned_by": "kokoro"
            },
            "tts-1-hd": {
                "id": "tts-1-hd",
                "object": "model",
                "created": 1686935002,
                "owned_by": "kokoro"
            },
            "kokoro": {
                "id": "kokoro",
                "object": "model",
                "created": 1686935002,
                "owned_by": "kokoro"
            }
        }
        
        # Check if requested model exists
        if model not in models:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "model_not_found",
                    "message": f"Model '{model}' not found",
                    "type": "invalid_request_error"
                }
            )
        
        # Return the specific model
        return models[model]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving model {model}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "server_error",
                "message": "Failed to retrieve model information",
                "type": "server_error",
            },
        )

@router.get("/audio/voices")
async def list_voices():
    """List all available voices for text-to-speech"""
    try:
        tts_service = await get_tts_service()
        voices = await tts_service.list_voices()
        return {"voices": voices}
    except Exception as e:
        logger.error(f"Error listing voices: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "server_error",
                "message": "Failed to retrieve voice list",
                "type": "server_error",
            },
        )


@router.post("/audio/voices/combine")
async def combine_voices(request: Union[str, List[str]]):
    """Combine multiple voices into a new voice and return the .pt file.

    Args:
        request: Either a string with voices separated by + (e.g. "voice1+voice2")
                or a list of voice names to combine

    Returns:
        FileResponse with the combined voice .pt file

    Raises:
        HTTPException:
            - 400: Invalid request (wrong number of voices, voice not found)
            - 500: Server error (file system issues, combination failed)
    """
    # Check if local voice saving is allowed
    if not settings.allow_local_voice_saving:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "permission_denied",
                "message": "Local voice saving is disabled",
                "type": "permission_error",
            },
        )

    try:
        # Convert input to list of voices
        if isinstance(request, str):
            # Check if it's an OpenAI voice name
            mapped_voice = _openai_mappings["voices"].get(request)
            if mapped_voice:
                request = mapped_voice
            voices = [v.strip() for v in request.split("+") if v.strip()]
        else:
            # For list input, map each voice if it's an OpenAI voice name
            voices = [_openai_mappings["voices"].get(v, v) for v in request]
            voices = [v.strip() for v in voices if v.strip()]

        if not voices:
            raise ValueError("No voices provided")

        # For multiple voices, validate base voices exist
        tts_service = await get_tts_service()
        available_voices = await tts_service.list_voices()
        for voice in voices:
            if voice not in available_voices:
                raise ValueError(
                    f"Base voice '{voice}' not found. Available voices: {', '.join(sorted(available_voices))}"
                )

        # Combine voices
        combined_tensor = await tts_service.combine_voices(voices=voices)
        combined_name = "+".join(voices)

        # Save to temp file
        temp_dir = tempfile.gettempdir()
        voice_path = os.path.join(temp_dir, f"{combined_name}.pt")
        buffer = io.BytesIO()
        torch.save(combined_tensor, buffer)
        async with aiofiles.open(voice_path, "wb") as f:
            await f.write(buffer.getvalue())

        return FileResponse(
            voice_path,
            media_type="application/octet-stream",
            filename=f"{combined_name}.pt",
            headers={
                "Content-Disposition": f"attachment; filename={combined_name}.pt",
                "Cache-Control": "no-cache",
            },
        )

    except ValueError as e:
        logger.warning(f"Invalid voice combination request: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_error",
                "message": str(e),
                "type": "invalid_request_error",
            },
        )
    except RuntimeError as e:
        logger.error(f"Voice combination processing error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": "Failed to process voice combination request",
                "type": "server_error",
            },
        )
    except Exception as e:
        logger.error(f"Unexpected error in voice combination: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "server_error",
                "message": "An unexpected error occurred",
                "type": "server_error",
            },
        )
