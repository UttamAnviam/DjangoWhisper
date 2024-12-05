# Generated by Django 5.1.3 on 2024-12-03 07:52

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='AudioFile',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('audio_name', models.CharField(max_length=255)),
                ('uuid', models.TextField(unique=True)),
                ('audio_size', models.FloatField(help_text='Size of the audio file in MB')),
                ('datetime', models.DateTimeField(auto_now_add=True)),
                ('status', models.CharField(choices=[('pending', 'Pending'), ('processing', 'Processing'), ('completed', 'Completed'), ('failed', 'Failed')], default='pending', max_length=20)),
            ],
        ),
    ]