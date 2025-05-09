from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
from django.conf import settings
import os
#import from core needs to be updated to coreapi and functions need to be fixed post migration 
# Add proper utils.py in the coreapi directory
from core.utils import process_directory_to_vectors, generate_embeddings_together, chat_with_llm_together
from django.http import JsonResponse
import threading
from .models import Document
from .serializers import DocumentSerializer

@api_view(["GET"])
def root_view(request):
    return Response({"message": "Welcome to MT Vector Knowledge Base"})

class ProcessDirectoryView(APIView):
    def post(self, request):
        directory_path = request.data.get("directory_path")
        chunk_size = int(request.data.get("chunk_size", 1000))
        chunk_overlap = int(request.data.get("chunk_overlap", 200))
        background = request.data.get("background", False)
        db_url = os.getenv("DATABASE_URL", getattr(settings, "DATABASE_URL", None))

        if not directory_path or not os.path.isdir(directory_path):
            return Response({"detail": f"Directory not found: {directory_path}"}, status=status.HTTP_400_BAD_REQUEST)

        def process_bg():
            try:
                process_directory_to_vectors(
                    directory_path=directory_path,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    db_session=None  # Update with Django ORM session 
                )
            except Exception as e:
                print(f"Background processing error: {str(e)}")

        try:
            if background:
                threading.Thread(target=process_bg).start()
                return Response({
                    "message": f"Processing directory in background: {directory_path}",
                    "status": "processing"
                }, status=status.HTTP_202_ACCEPTED)
            else:
                document_ids = process_directory_to_vectors(
                    directory_path=directory_path,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    db_session=None  # Replace with Django ORM session if needed
                )
                return Response({
                    "message": f"Successfully processed directory: {directory_path}",
                    "document_count": len(document_ids),
                    "document_ids": document_ids
                }, status=status.HTTP_200_OK)
        except ValueError as e:
            return Response({"detail": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({"detail": f"Error processing directory: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class GenerateEmbeddingsView(APIView):
    def post(self, request):
        text = request.data.get("text")
        model = request.data.get("model")
        if not text:
            return Response({"detail": "Text is required."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            embedding = generate_embeddings_together(text, model) if model else generate_embeddings_together(text)
            return Response({"embedding": embedding, "dimensions": len(embedding), "model": model})
        except ValueError as e:
            return Response({"detail": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({"detail": f"Error generating embeddings: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ChatView(APIView):
    def post(self, request):
        messages = request.data.get("messages")
        model = request.data.get("model")
        temperature = float(request.data.get("temperature", 0.7))
        max_tokens = int(request.data.get("max_tokens", 1000))
        if not messages:
            return Response({"detail": "Messages are required."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            if model:
                response = chat_with_llm_together(messages, model, temperature, max_tokens)
            else:
                response = chat_with_llm_together(messages, temperature=temperature, max_tokens=max_tokens)
            return Response({"response": response})
        except ValueError as e:
            return Response({"detail": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({"detail": f"Error chatting with LLM: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class DocumentListView(APIView):
    def get(self, request):
        limit = int(request.query_params.get("limit", 10))
        offset = int(request.query_params.get("offset", 0))
        documents = Document.objects.all()[offset:offset+limit]
        serializer = DocumentSerializer(documents, many=True)
        return Response(serializer.data)

class DocumentDetailView(APIView):
    def get(self, request, document_id):
        try:
            document = Document.objects.get(pk=document_id)
        except Document.DoesNotExist:
            return Response({"detail": f"Document with ID {document_id} not found"}, status=status.HTTP_404_NOT_FOUND)
        serializer = DocumentSerializer(document)
        return Response(serializer.data)
