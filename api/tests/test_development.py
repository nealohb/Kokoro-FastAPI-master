import pytest
from unittest.mock import patch, MagicMock
import requests

def test_generate_captioned_speech():
    """Test the generate_captioned_speech function with mocked responses"""
    # Mock the API responses
    mock_audio_response = MagicMock()
    mock_audio_response.status_code = 200
    mock_audio_response.content = b"mock audio data"
    mock_audio_response.headers = {"X-Timestamps-Path": "test.json"}

    mock_timestamps_response = MagicMock()
    mock_timestamps_response.status_code = 200
    mock_timestamps_response.json.return_value = [
        {"word": "test", "start_time": 0.0, "end_time": 1.0}
    ]

    # Patch both HTTP requests
    with patch('requests.post', return_value=mock_audio_response), \
         patch('requests.get', return_value=mock_timestamps_response):
        
        # Import here to avoid module-level import issues
        from examples.captioned_speech_example import generate_captioned_speech
        
        # Test the function
        audio, timestamps = generate_captioned_speech("test text")
        
        # Verify we got both audio and timestamps
        assert audio == b"mock audio data"
        assert timestamps == [{"word": "test", "start_time": 0.0, "end_time": 1.0}]