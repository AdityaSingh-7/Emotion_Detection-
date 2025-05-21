from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
import os
import cv2
import torch 
import numpy as np
import subprocess
import io
import array
import tempfile

class MELDDataset(Dataset):
    def __init__(self, csv_path, video_dir):
        self.data = pd.read_csv(csv_path)
        self.video_dir = video_dir

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        self.emotion_map = {
            'anger': 0,
            'disgust': 1,
            'fear': 2,
            'joy': 3,
            'neutral': 4,
            'sadness': 5,
            'surprise': 6,
        }

        self.sentiment_map = {
            'negative': 0,
            'neutral': 1,
            'positive': 2,
        }

    def load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        try:
            if not cap.isOpened():
                raise FileNotFoundError(f"File {video_path} not found")
            
            ret, frame = cap.read()
            if not ret or frame is None:    
                raise ValueError(f"Cannot read frame from {video_path}")
            
            # Reset the video capture to the beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            while len(frames) < 30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frame = cv2.resize(frame, (224, 224))
                frame = frame/255.0
                frames.append(frame)
            
        except Exception as e:
            raise ValueError(f"Error reading video {video_path}: {e}")
        finally:    
            cap.release()

        if len(frames) == 0:
            raise ValueError(f"Video {video_path} has no frames")
        
        # Pad or truncate frames
        if len(frames) < 30:
            frames += [np.zeros_like(frames[0])] * (30 - len(frames))
        else:    
            frames = frames[:30]

        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)
    
    def extract_audio_features(self, video_path):
        """Extract audio features using raw PCM data and native Python processing"""
        temp_audio_path = None
        
        try:
            # Create a temporary file for the audio
            temp_fd, temp_audio_path = tempfile.mkstemp(suffix='.raw')
            os.close(temp_fd)
            
            # Extract audio using ffmpeg to raw PCM format
            command = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', '16000',  # 16kHz sampling rate
                '-ac', '1',  # Mono
                '-f', 's16le',  # Raw PCM format
                temp_audio_path
            ]
            
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Read raw PCM data
            with open(temp_audio_path, 'rb') as f:
                pcm_data = f.read()
                
            # Convert to numpy array
            audio_array = np.frombuffer(pcm_data, dtype=np.int16)
            
            # Normalize to [-1, 1]
            audio_array = audio_array.astype(np.float32) / 32768.0
            
            # Compute features (simplified mel-like spectrogram)
            feature_tensor = self._compute_spectrogram(audio_array, sample_rate=16000)
            
            return feature_tensor
            
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e}")
            raise ValueError(f"Error extracting audio: {e}")
        except Exception as e:
            print(f"Audio processing error: {e}")
            raise ValueError(f"Error processing audio: {e}")
        finally:
            # Clean up temporary file
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
    
    def _compute_spectrogram(self, audio_array, sample_rate=16000):
        """Compute a spectrogram-like representation using only numpy"""
        # Parameters
        n_fft = 1024
        hop_length = 512
        n_mels = 64
        max_frames = 300
        
        # Calculate number of frames
        n_frames = 1 + (len(audio_array) - n_fft) // hop_length
        n_frames = min(n_frames, max_frames)
        
        # Initialize feature array
        spectrogram = np.zeros((n_mels, n_frames))
        
        # Window function (Hann)
        window = np.hanning(n_fft)
        
        # Process each frame
        for i in range(n_frames):
            # Extract frame
            start = i * hop_length
            end = start + n_fft
            
            # Apply window and compute FFT
            if end <= len(audio_array):
                frame = audio_array[start:end] * window
                # Compute power spectrum
                fft_result = np.abs(np.fft.rfft(frame))**2
                
                # Simple mel-like scaling (just take the first n_mels bins or downsample)
                if len(fft_result) >= n_mels:
                    # Simple downsampling by taking every nth element
                    indices = np.linspace(0, len(fft_result)-1, n_mels, dtype=int)
                    spectrogram[:, i] = fft_result[indices]
                else:
                    # If we have fewer FFT bins than mels, just repeat values
                    spectrogram[:, i] = np.repeat(fft_result, n_mels // len(fft_result) + 1)[:n_mels]
        
        # Apply log scaling
        spectrogram = np.log(spectrogram + 1e-9)
        
        # Normalize
        if np.std(spectrogram) > 0:
            spectrogram = (spectrogram - np.mean(spectrogram)) / np.std(spectrogram)
        
        # Pad if needed
        if spectrogram.shape[1] < max_frames:
            padding = max_frames - spectrogram.shape[1]
            spectrogram = np.pad(spectrogram, ((0, 0), (0, padding)))
        
        # Convert to tensor
        return torch.FloatTensor(spectrogram).unsqueeze(0)  # Add channel dimension
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
        video_path = os.path.join(self.video_dir, video_filename)

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"File {video_path} not found")
        
        text_inputs = self.tokenizer(
            row['Utterance'],
            padding="max_length",
            truncation=True,
            max_length=128, 
            return_tensors='pt'
        )
        
        # Remove the batch dimension from tokenizer output
        for key in text_inputs:
            text_inputs[key] = text_inputs[key].squeeze(0)
        
        # Extract features
        video_frames = self.load_video_frames(video_path)
        audio_features = self.extract_audio_features(video_path)
        
        # Get labels
        emotion_label = torch.tensor(self.emotion_map[row['Emotion']])
        sentiment_label = torch.tensor(self.sentiment_map[row['Sentiment']])
        
        return {
            'text': text_inputs,
            'video': video_frames,
            'audio': audio_features,
            'emotion': emotion_label,
            'sentiment': sentiment_label,
            'dialogue_id': row['Dialogue_ID'],
            'utterance_id': row['Utterance_ID']
        }

if __name__ == "__main__":
    # Update these paths to match your actual file locations
    csv_path = r'C:/Users/Aditya Singh/Desktop/Emotion_Detection/dataset/dev/dev_sent_emo.csv'
    video_dir = r'C:/Users/Aditya Singh/Desktop/Emotion_Detection/dataset/dev/dev_splits_complete'
    
    print("Creating MELD dataset...")
    meld = MELDDataset(csv_path, video_dir)
    print(f"Dataset created with {len(meld)} samples")
    
    print("Testing first sample extraction...")
    sample = meld[0]
    
    print(f"Text shape: {sample['text']['input_ids'].shape}")
    print(f"Video shape: {sample['video'].shape}")
    print(f"Audio shape: {sample['audio'].shape}")
    print(f"Emotion: {sample['emotion'].item()}")
    print(f"Sentiment: {sample['sentiment'].item()}")
    print(f"Dialogue ID: {sample['dialogue_id']}")
    print(f"Utterance ID: {sample['utterance_id']}")
    print("Success!")