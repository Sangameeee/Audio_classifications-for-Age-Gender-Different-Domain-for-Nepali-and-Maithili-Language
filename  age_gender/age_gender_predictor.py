import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
import librosa
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")


class ModelHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        if not hasattr(self, 'all_tied_weights_keys'):
            self.all_tied_weights_keys = defaultdict(list)

        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)
        return hidden_states, logits_age, logits_gender


def load_audio(file_path, target_sr=16000):
    """Load audio file (.wav or .mp3) and resample to 16kHz mono."""
    try:
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        return audio, target_sr
    except Exception as e:
        raise RuntimeError(f"Error loading audio file '{file_path}': {str(e)}")


class AgeGenderPredictor:
    """
    Efficient predictor for age/gender inference with model loaded once.
    """
    def __init__(self, model_name="audeering/wav2vec2-large-robust-24-ft-age-gender", device=None):
        """
        Initialize predictor with model loaded once.

        Args:
            model_name (str): Hugging Face model identifier
            device (str, optional): 'cuda' or 'cpu'. Auto-detected if None.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[AgeGenderPredictor] Loading model '{model_name}' on {self.device}...", flush=True)
        
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = AgeGenderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Disable gradient tracking for inference
        for param in self.model.parameters():
            param.requires_grad = False
            
        print(f"[AgeGenderPredictor] Model loaded successfully ✓")

    def predict(self, audio_path):
        """
        Predict age and gender from audio file.

        Args:
            audio_path (str): Path to .wav or .mp3 file

        Returns:
            dict: {
                'age': float (0-100),
                'gender': str ('female'/'male'/'child'),
                'gender_confidence': float (0-1),
                'gender_distribution': dict with probabilities per class
            }
        """
        # Load and preprocess audio
        audio, sr = load_audio(audio_path)
        input_values = self.processor(
            audio, 
            sampling_rate=sr, 
            return_tensors="pt"
        ).input_values.to(self.device)

        # Run inference
        with torch.no_grad():
            _, logits_age, probs_gender = self.model(input_values)

        # Extract predictions
        age = logits_age.item() * 100  # Scale to [0, 100] years
        gender_probs = probs_gender[0].cpu().numpy()
        gender_classes = ['female', 'male', 'child']
        gender_pred = gender_classes[np.argmax(gender_probs)]

        return {
            'age': round(age, 1),
            'gender': gender_pred,
            'gender_confidence': round(float(np.max(gender_probs)), 3),
            'gender_distribution': {
                'female': round(float(gender_probs[0]), 3),
                'male': round(float(gender_probs[1]), 3),
                'child': round(float(gender_probs[2]), 3)
            }
        }

    def predict_batch(self, audio_paths):
        """
        Predict age/gender for multiple audio files efficiently.

        Args:
            audio_paths (list[str]): List of audio file paths

        Returns:
            list[dict]: List of prediction results in same order as input
        """
        return [self.predict(path) for path in audio_paths]


def display_results(result, audio_path=None):
    """Pretty-print prediction results."""
    print("\n" + "="*60)
    print("AGE AND GENDER PREDICTION RESULTS")
    print("="*60)
    if audio_path:
        print(f"Audio File: {audio_path}")
    print(f"Predicted Age: {result['age']} years")
    print(f"Predicted Gender: {result['gender'].upper()} "
          f"(confidence: {result['gender_confidence']:.1%})")
    print("\nGender Probability Distribution:")
    for gender, prob in result['gender_distribution'].items():
        bar = "█" * int(prob * 40)
        print(f"  {gender:6s}: {bar:40s} {prob:.1%}")
    print("="*60 + "\n")