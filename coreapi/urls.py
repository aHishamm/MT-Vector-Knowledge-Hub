from django.urls import path
from .views import (
    root_view,
    ProcessDirectoryView,
    GenerateEmbeddingsView,
    ChatView,
    DocumentListView,
    DocumentDetailView,
)

urlpatterns = [
    path('', root_view, name='root'),
    path('process-directory/', ProcessDirectoryView.as_view(), name='process-directory'),
    path('generate-embeddings/', GenerateEmbeddingsView.as_view(), name='generate-embeddings'),
    path('chat/', ChatView.as_view(), name='chat'),
    path('documents/', DocumentListView.as_view(), name='documents-list'),
    path('documents/<int:document_id>/', DocumentDetailView.as_view(), name='documents-detail'),
]
