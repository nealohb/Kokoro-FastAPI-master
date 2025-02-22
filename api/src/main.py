"""
FastAPI OpenAI Compatible API
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stdout, level="DEBUG")

app = FastAPI(
    title="Kokoro TTS API",
    description="API for text-to-speech generation using Kokoro",
    version="0.1.4",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.debug("Health check called")
    return {"status": "healthy"}

@app.get("/")
async def root():
    """Root endpoint"""
    logger.debug("Root endpoint called")
    return {"message": "Kokoro TTS API is running"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000))) 