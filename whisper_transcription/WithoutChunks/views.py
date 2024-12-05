import logging
import os
import asyncio
import httpx
from django.http import JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from asgiref.sync import sync_to_async
from .audio_transcription_manager import AudioTranscriptionManager
from django.utils.decorators import method_decorator
from .models import AudioFile
from django.conf import settings

# Initialize the transcription manager
transcription_manager = AudioTranscriptionManager()

@method_decorator(csrf_exempt, name="dispatch")
class NoChunksTranscribeView(View):
    async def post(self, request):
        url = request.POST.get("url")
        uuid_param = request.POST.get("uuid")

        if not url or not uuid_param:
            return JsonResponse({"detail": "URL and UUID are required"}, status=400)

        try:
            # Create audio file record
            audio_file = await sync_to_async(self.create_audio_file)(os.path.basename(url), uuid_param, status='pending')

            # Process the request asynchronously
            result = await self.process_request(url, uuid_param)

            return JsonResponse({
                "results": [{
                    "filename": os.path.basename(url),
                    "transcript": {
                        "text": result['transcript']['text'],  # Actual transcription text
                        "segments": result['transcript'].get('segments', []),  # Segments if available
                        "language": result['transcript'].get('language', "ml")  # Language info
                    },
                    "uuid": uuid_param,
                    "status": "completed"
                }]
            })

        except asyncio.TimeoutError:
            await self.update_audio_file_status(uuid_param, 'timeout')
            return JsonResponse({"detail": "Request timed out"}, status=408)

        except Exception as e:
            logging.error(f"Error processing request: {e}")
            await self.update_audio_file_status(uuid_param, 'error')
            return JsonResponse({"detail": "Internal server error"}, status=500)

    async def process_request(self, url, uuid_param):
        async with httpx.AsyncClient() as client:
            temp_file_path = f"/tmp/{uuid_param}"

            try:
                # Download the audio file
                response = await client.get(url)
                response.raise_for_status()

                # Save the file locally
                with open(temp_file_path, "wb") as f:
                    f.write(response.content)

                # Update status in database to "processing"
                await self.update_audio_file_status(uuid_param, 'processing')

                # Perform transcription
                result = await asyncio.wait_for(
                    transcription_manager.transcribe_audio(temp_file_path),
                    timeout=settings.TIMEOUT_SECONDS
                )

                if result is None:
                    raise Exception("Transcription failed")

                # Ensure the transcription text is present
                transcription_result = {
                    'text': result.get('text', 'Transcription failed'),  # Actual transcription text
                    'language': result.get('language', 'ml')  # Language info
                }

                await self.update_audio_file_status(uuid_param, 'completed')

                return {'transcript': transcription_result, 'status': 'completed'}

            finally:
                # Cleanup the temp file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

    @staticmethod
    async def update_audio_file_status(uuid_param, status):
        try:
            audio_file = await sync_to_async(AudioFile.objects.get)(uuid=uuid_param)
            audio_file.status = status
            await sync_to_async(audio_file.save)()
        except AudioFile.DoesNotExist:
            logging.error(f"Audio file with UUID {uuid_param} does not exist.")
        except Exception as e:
            logging.error(f"Error updating audio file status: {e}")

    @staticmethod
    def create_audio_file(filename, uuid_param, status='pending'):
        try:
            return AudioFile.objects.create(audio_name=filename, uuid=uuid_param, status=status)
        except Exception as e:
            logging.error(f"Error creating audio file record: {e}")
            return None
