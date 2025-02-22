"""Temporary file management for audio files"""

import os
import asyncio
import aiofiles
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from loguru import logger

class TempFileWriter:
    """Async context manager for writing temporary audio files"""
    
    def __init__(self, format: str, base_dir: Optional[str] = None):
        self.format = format
        self.base_dir = base_dir or "/tmp/kokoro_audio"
        self._file = None
        self._finalized = False
        self.file_path = None
        self.download_path = None
        
    async def __aenter__(self):
        """Create temp file and return self"""
        # Ensure base directory exists
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"speech_{timestamp}_{os.urandom(4).hex()}.{self.format}"
        self.file_path = os.path.join(self.base_dir, filename)
        
        # Create download path
        self.download_path = f"/download/{filename}"
        
        # Open file for writing
        self._file = await aiofiles.open(self.file_path, "wb")
        return self
        
    async def write(self, data: bytes):
        """Write data to temp file"""
        if self._file and not self._file.closed:
            await self._file.write(data)
            await self._file.flush()  # Ensure data is written to disk
            
    async def finalize(self):
        """Finalize the temp file"""
        if self._file and not self._file.closed:
            await self._file.flush()
            await self._file.close()
            self._finalized = True
            
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close file if not already closed"""
        if self._file and not self._file.closed:
            await self._file.close()
            self._finalized = True

async def cleanup_temp_files():
    """Clean up old temporary files"""
    temp_dir = "/tmp/kokoro_audio"
    if not os.path.exists(temp_dir):
        return
        
    try:
        # Get current time
        now = datetime.now()
        
        # List all files in temp directory
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            
            try:
                # Get file stats
                stats = os.stat(file_path)
                file_time = datetime.fromtimestamp(stats.st_mtime)
                
                # Remove files older than 1 hour
                if now - file_time > timedelta(hours=1):
                    os.remove(file_path)
                    logger.debug(f"Removed old temp file: {filename}")
                    
            except Exception as e:
                logger.error(f"Error processing temp file {filename}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error cleaning up temp files: {e}")

# Schedule periodic cleanup
async def schedule_cleanup(interval_seconds: int = 300):  # 5 minutes
    """Schedule periodic cleanup of temp files"""
    while True:
        try:
            await cleanup_temp_files()
        except Exception as e:
            logger.error(f"Error in scheduled cleanup: {e}")
        await asyncio.sleep(interval_seconds)
