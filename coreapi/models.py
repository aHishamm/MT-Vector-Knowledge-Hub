from django.db import models

class Document(models.Model):
    title = models.CharField(max_length=255)
    content = models.TextField()
    embedding = models.JSONField(null=True, blank=True)  #Change to store as a vector later (needs updating)

    def __str__(self):
        return self.title
