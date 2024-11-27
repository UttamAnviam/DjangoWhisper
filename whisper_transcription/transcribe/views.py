import logging
import os
import asyncio
from tempfile import NamedTemporaryFile
from concurrent.futures import ThreadPoolExecutor
from pydub import AudioSegment
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
import torch
import whisper
from transformers import pipeline
import httpx
import time

# Set up logging
logging.basicConfig(level=logging.INFO)

# Constants
CHUNK_SIZE_MB = 25
EXPECTED_TOKEN = "NgPEbNQnZrKDtRwfaIrBmnryRQZITFhm"
CHUNK_DURATION_MS = 180000

# Initialize models
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# Initialize models with consistent configuration
whisper_model = whisper.load_model("medium").to(device)
pipeline_model = whisper.load_model("medium").to(device)  # Using Whisper directly instead of pipeline

# ThreadPoolExecutor for managing synchronous tasks
executor = ThreadPoolExecutor()

def split_audio_into_chunks(file_path, chunk_duration_ms=CHUNK_DURATION_MS):
    """Split audio into chunks and return the file paths of each chunk."""
    audio = AudioSegment.from_file(file_path)
    chunks = []
    for i in range(0, len(audio), chunk_duration_ms):
        chunk = audio[i:i + chunk_duration_ms]
        temp_chunk = NamedTemporaryFile(delete=False, suffix=".wav")
        chunk.export(temp_chunk.name, format="wav")
        chunks.append((temp_chunk.name, i / 1000))  # Start time in seconds
    return chunks

async def transcribe_chunk(chunk_info):
    """Transcribe a single chunk using Whisper model directly."""
    chunk_path, chunk_start_time = chunk_info
    start_time = time.time()
    try:
        # Use Whisper model directly instead of pipeline
        result = whisper_model.transcribe(chunk_path)
        end_time = time.time()

        # Adjust segment timestamps
        adjusted_segments = []
        if 'segments' in result:
            for segment in result['segments']:
                adjusted_segment = segment.copy()
                adjusted_segment['start'] = segment['start'] + chunk_start_time
                adjusted_segment['end'] = segment['end'] + chunk_start_time
                adjusted_segments.append(adjusted_segment)

        return {
            'text': result['text'],
            'segments': adjusted_segments,
            'start_time': start_time,
            'end_time': end_time
        }
    except Exception as e:
        logging.error(f"Transcription error for {chunk_path}: {e}")
        return None
    finally:
        try:
            os.remove(chunk_path)
        except Exception as e:
            logging.error(f"Error removing temporary file {chunk_path}: {e}")

async def process_audio(file_path):
    """Process large audio files by splitting into chunks."""
    chunks = split_audio_into_chunks(file_path)
    tasks = [transcribe_chunk(chunk) for chunk in chunks]
    chunk_results = await asyncio.gather(*tasks)

    transcriptions = []
    all_segments = []
    processing_times = []

    for result in chunk_results:
        if result:
            transcriptions.append(result['text'])
            if 'segments' in result and result['segments']:
                all_segments.extend(result['segments'])
            processing_times.append({
                'start_time': result['start_time'],
                'end_time': result['end_time']
            })

    # Sort segments by start time
    all_segments.sort(key=lambda x: x['start'])

    return {
        'text': " ".join(transcriptions),
        'language': "en",
        'segments': all_segments,  # This will now have the same format as whisper output
        'processing_times': processing_times
    }

def transcribe_whisper(file_path):
    """Transcribe using Whisper for smaller files."""
    start_time = time.time()
    result = whisper_model.transcribe(file_path)
    end_time = time.time()

    return {
        'text': result['text'],
        'language': result['language'],
        'segments': result['segments'],
        'processing_times': [{
            'start_time': start_time,
            'end_time': end_time
        }]
    }

@method_decorator(csrf_exempt, name="dispatch")
class TranscribeView(View):
    async def post(self, request):
        token = request.headers.get("X-Token")
        if token != EXPECTED_TOKEN:
            return JsonResponse({"detail": "Forbidden: Invalid token"}, status=403)

        url = request.POST.get("url")
        if not url:
            return JsonResponse({"detail": "URL is required"}, status=400)

        filename = os.path.basename(url)
        temp_file_path = f"/tmp/{filename}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                if response.status_code != 200:
                    return JsonResponse({"detail": "Failed to download file"}, status=400)

            with open(temp_file_path, "wb") as file:
                file.write(response.content)

            file_size_mb = os.path.getsize(temp_file_path) / (1024 * 1024)
            logging.info(f"Downloaded file {filename} ({file_size_mb:.2f} MB)")

            if file_size_mb > CHUNK_SIZE_MB:
                result = await process_audio(temp_file_path)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(executor, transcribe_whisper, temp_file_path)

            download_url = f"https://process-audio.healthorbit.ai/transcription-results/{filename}"

            return JsonResponse({
                "results": [{
                    "filename": filename,
                    "transcript": {
                        "text": result['text'],
                        "segments": result['segments'],
                        "language": result['language']
                    },
                    "download_url": download_url,
                    "processing_times": result['processing_times']
                }]
            })

        except Exception as e:
            logging.error(f"Error during file processing: {e}")
            return JsonResponse({"detail": "Internal server error"}, status=500)

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)