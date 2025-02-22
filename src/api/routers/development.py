from typing import List, Union, AsyncGenerator, Tuple

import numpy as np
import torch
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse, FileResponse
from kokoro import KPipeline
from loguru import logger

from ..core.config import settings
from ..services.audio import AudioNormalizer, AudioService
from ..services.streaming_audio_writer import StreamingAudioWriter
from ..services.text_processing import smart_split
from ..services.tts_service import TTSService
from ..services.temp_manager import TempFileWriter
from ..structures import CaptionedSpeechRequest, CaptionedSpeechResponse, WordTimestamp
from ..structures.text_schemas import (
    GenerateFromPhonemesRequest,
    PhonemeRequest,
    PhonemeResponse,
)
import json
import os
from pathlib import Path


router = APIRouter(tags=["text processing"])


async def get_tts_service() -> TTSService:
    """Dependency to get TTSService instance"""
    return (
        await TTSService.create()
    )  # Create service with properly initialized managers


@router.post("/dev/phonemize", response_model=PhonemeResponse)
async def phonemize_text(request: PhonemeRequest) -> PhonemeResponse:
    """Convert text to phonemes using Kokoro's quiet mode.

    Args:
        request: Request containing text and language

    Returns:
        Phonemes and token IDs
    """
    try:
        if not request.text:
            raise ValueError("Text cannot be empty")

        # Initialize Kokoro pipeline in quiet mode (no model)
        pipeline = KPipeline(lang_code=request.language, model=False)

        # Get first result from pipeline (we only need one since we're not chunking)
        for result in pipeline(request.text):
            # result.graphemes = original text
            # result.phonemes = phonemized text
            # result.tokens = token objects (if available)
            return PhonemeResponse(phonemes=result.phonemes, tokens=[])

        raise ValueError("Failed to generate phonemes")
    except ValueError as e:
        logger.error(f"Error in phoneme generation: {str(e)}")
        raise HTTPException(
            status_code=500, detail={"error": "Server error", "message": str(e)}
        )
    except Exception as e:
        logger.error(f"Error in phoneme generation: {str(e)}")
        raise HTTPException(
            status_code=500, detail={"error": "Server error", "message": str(e)}
        )


@router.post("/dev/generate_from_phonemes")
async def generate_from_phonemes(
    request: GenerateFromPhonemesRequest,
    client_request: Request,
    tts_service: TTSService = Depends(get_tts_service),
) -> StreamingResponse:
    """Generate audio directly from phonemes using Kokoro's phoneme format"""
    try:
        # Basic validation
        if not isinstance(request.phonemes, str):
            raise ValueError("Phonemes must be a string")
        if not request.phonemes:
            raise ValueError("Phonemes cannot be empty")

        # Create streaming audio writer and normalizer
        writer = StreamingAudioWriter(format="wav", sample_rate=24000, channels=1)
        normalizer = AudioNormalizer()

        async def generate_chunks():
            try:
                # Generate audio from phonemes
                chunk_audio, _ = await tts_service.generate_from_phonemes(
                    phonemes=request.phonemes,  # Pass complete phoneme string
                    voice=request.voice,
                    speed=1.0,
                )

                if chunk_audio is not None:
                    # Normalize audio before writing
                    normalized_audio = await normalizer.normalize(chunk_audio)
                    # Write chunk and yield bytes
                    chunk_bytes = writer.write_chunk(normalized_audio)
                    if chunk_bytes:
                        yield chunk_bytes

                    # Finalize and yield remaining bytes
                    final_bytes = writer.write_chunk(finalize=True)
                    if final_bytes:
                        yield final_bytes
                else:
                    raise ValueError("Failed to generate audio data")

            except Exception as e:
                logger.error(f"Error in audio generation: {str(e)}")
                # Clean up writer on error
                writer.write_chunk(finalize=True)
                # Re-raise the original exception
                raise

        return StreamingResponse(
            generate_chunks(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav",
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "Transfer-Encoding": "chunked",
            },
        )

    except ValueError as e:
        logger.error(f"Error generating audio: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_error",
                "message": str(e),
                "type": "invalid_request_error",
            },
        )
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": str(e),
                "type": "server_error",
            },
        )


@router.get("/dev/timestamps/{filename}")
async def get_timestamps(filename: str):
    """Download timestamps from temp storage"""
    try:
        from ..core.paths import _find_file

        # Search for file in temp directory
        file_path = await _find_file(
            filename=filename, search_paths=[settings.temp_file_dir]
        )

        return FileResponse(
            file_path,
            media_type="application/json",
            filename=filename,
            headers={
                "Cache-Control": "no-cache",
                "Content-Disposition": f"attachment; filename={filename}",
            },
        )

    except Exception as e:
        logger.error(f"Error serving timestamps file {filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "server_error",
                "message": "Failed to serve timestamps file",
                "type": "server_error",
            },
        )


@router.post("/dev/captioned_speech")
async def create_captioned_speech(
    request: CaptionedSpeechRequest,
    client_request: Request,
    x_raw_response: str = Header(None, alias="x-raw-response"),
    tts_service: TTSService = Depends(get_tts_service),
):
    """Generate audio with word-level timestamps using streaming approach"""
    try:
        # Set content type based on format
        content_type = {
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav",
            "pcm": "audio/pcm",
        }.get(request.response_format, f"audio/{request.response_format}")

        # Create streaming audio writer and normalizer
        writer = StreamingAudioWriter(
            format=request.response_format, sample_rate=24000, channels=1
        )
        normalizer = AudioNormalizer()

        # Get voice path
        voice_name, voice_path = await tts_service._get_voice_path(request.voice)

        # Use provided lang_code or determine from voice name
        pipeline_lang_code = request.lang_code if request.lang_code else request.voice[0].lower()
        logger.info(
            f"Using lang_code '{pipeline_lang_code}' for voice '{voice_name}' in text chunking"
        )

        # Get backend and pipeline
        backend = tts_service.model_manager.get_backend()
        pipeline = backend._get_pipeline(pipeline_lang_code)

        # Create temp file writer for timestamps
        temp_writer = TempFileWriter("json")
        await temp_writer.__aenter__()  # Initialize temp file
        # Get just the filename without the path
        timestamps_filename = Path(temp_writer.download_path).name

        # Initialize variables for timestamps
        word_timestamps = []
        current_offset = 0.0

        async def generate_chunks():
            nonlocal current_offset, word_timestamps
            try:
                # Process text in chunks with smart splitting
                async for chunk_text, tokens in smart_split(request.input):
                    # Process chunk with pipeline
                    for result in pipeline(chunk_text, voice=voice_path, speed=request.speed):
                        if result.audio is not None:
                            # Process timestamps for this chunk
                            if hasattr(result, "tokens") and result.tokens and result.pred_dur is not None:
                                try:
                                    # Join timestamps for this chunk's tokens
                                    KPipeline.join_timestamps(result.tokens, result.pred_dur)

                                    # Add timestamps with offset
                                    for token in result.tokens:
                                        if not all(
                                            hasattr(token, attr)
                                            for attr in ["text", "start_ts", "end_ts"]
                                        ):
                                            continue
                                        if not token.text or not token.text.strip():
                                            continue

                                        # Apply offset to timestamps
                                        start_time = float(token.start_ts) + current_offset
                                        end_time = float(token.end_ts) + current_offset

                                        word_timestamps.append(
                                            {
                                                "word": str(token.text).strip(),
                                                "start_time": start_time,
                                                "end_time": end_time,
                                            }
                                        )

                                    # Update offset for next chunk
                                    chunk_duration = float(result.pred_dur.sum()) / 80  # Convert frames to seconds
                                    current_offset = max(current_offset + chunk_duration, end_time)

                                except Exception as e:
                                    logger.error(f"Failed to process timestamps for chunk: {e}")

                            # Process audio
                            audio_chunk = result.audio.numpy()
                            normalized_audio = await normalizer.normalize(audio_chunk)
                            chunk_bytes = writer.write_chunk(normalized_audio)
                            if chunk_bytes:
                                yield chunk_bytes

                # Write timestamps to temp file
                timestamps_json = json.dumps(word_timestamps)
                await temp_writer.write(timestamps_json.encode())
                await temp_writer.finalize()

                # Finalize audio
                final_bytes = writer.write_chunk(finalize=True)
                if final_bytes:
                    yield final_bytes

            except Exception as e:
                logger.error(f"Error in audio generation: {str(e)}")
                # Clean up writer on error
                writer.write_chunk(finalize=True)
                await temp_writer.__aexit__(type(e), e, e.__traceback__)
                # Re-raise the original exception
                raise

        return StreamingResponse(
            generate_chunks(),
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "Transfer-Encoding": "chunked",
                "X-Timestamps-Path": timestamps_filename,
            },
        )

    except ValueError as e:
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
        logger.error(f"Unexpected error in speech generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": str(e),
                "type": "server_error",
            },
        )
