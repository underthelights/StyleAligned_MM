import torchaudio
import torch
from transformers import ASTForAudioClassification, ASTFeatureExtractor

# Function to load and preprocess audio
def load_and_preprocess_audio(audio_path, target_sample_rate=16000):
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)  # Convert to mono
    return waveform, target_sample_rate

# Function to classify audio using AST and custom labels
def classify_audio_with_ast(audio_path, custom_labels):
    # Load AST model and feature extractor
    model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    
    # Load and preprocess audio
    waveform, sample_rate = load_and_preprocess_audio(audio_path)
    inputs = feature_extractor(waveform.numpy(), sampling_rate=sample_rate, return_tensors="pt", padding=True)

    # Get predictions
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    
    # Map prediction to custom labels
    id2label = model.config.id2label  # Predefined AST labels
    predefined_label = id2label.get(predicted_class_idx, "Unknown")
    
    # Custom mapping (example: predefined labels to "Cat", "Motorcycle", etc.)
    custom_mapping = {
        "Domestic animals": "Cat",
        "Vehicle": "Motorcycle",
        "Music": "Wolf",
        "Siren": "Ambulance",
    }
    custom_label = custom_mapping.get(predefined_label, "Other")
    
    print(f"Predicted Predefined Label: {predefined_label}")
    print(f"Mapped to Custom Label: {custom_label}")
    return custom_label

# Example Usage
# audio_path = "asset/audio_animal/mixkit-ambulance-siren-us-1642.wav"
# audio_path = "asset/audio_animal/mixkit-motocross-motorcycle-engine-2727.wav"
audio_path = "asset/audio_animal/mixkit-wolf-howling-1775.wav"
custom_labels = ["Cat", "Motorcycle", "Wolf", "Ambulance"]
classify_audio_with_ast(audio_path, custom_labels)
