import logging
import os
import httpx
from rq import Queue
from redis import Redis
from rq.job import Job
from .audio_transcription_manager import AudioTranscriptionManager
from .models import AudioFile

# Initialize Redis Queue
redis_conn = Redis()
queue = Queue(connection=redis_conn)

# Initialize Transcription Manager
transcription_manager = AudioTranscriptionManager()

def process_audio(url, uuid_param):
    temp_file_path = f"/tmp/{uuid_param}"

    try:
        # Download audio file
        response = httpx.get(url)
        response.raise_for_status()

        # Save the file
        with open(temp_file_path, "wb") as f:
            f.write(response.content)

        # Update status to processing
        audio_file = AudioFile.objects.get(uuid=uuid_param)
        audio_file.status = 'processing'
        audio_file.save()

        # Perform transcription
        result = transcription_manager._transcribe_sync(temp_file_path)

        if result is None:
            raise Exception("Transcription failed")

        # Update status to completed
        audio_file.status = 'completed'
        audio_file.save()

        return {'transcript': result['text'], 'status': 'completed'}

    finally:
        # Cleanup
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
