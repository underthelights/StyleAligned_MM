import torchaudio
import torch
import hashlib
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

# Function to generate a unique seed from the audio file
def generate_seed_from_audio(audio_path):
    with open(audio_path, 'rb') as f:
        audio_data = f.read()
    return int(hashlib.md5(audio_data).hexdigest(), 16) % (2**32)

# Function to generate music description based on Wav2CLIP
def generate_music_description_wav2clip(audio_path):
    # Generate a unique seed based on the audio file
    seed = generate_seed_from_audio(audio_path)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Load the Wav2CLIP model and processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base-960h")
    
    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Resample to 16kHz if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Make sure it's mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Preprocess audio
    inputs = processor(waveform.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)

    # Get predictions
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

    # Generate description (Wav2CLIP does not include a label set, so user-defined mapping is needed)
    labels = {0: "Cat", 1: "Motorcycle", 2: "Wolf", 3: "Ambulance"}  # Example label mapping
    music_description = labels.get(predicted_class_idx, "Unknown")
    
    print("Generated Music Description (Wav2CLIP):", music_description)
    return music_description

# Example Usage
audio_path = "asset/audio_animal/mixkit-ambulance-siren-us-1642.wav"
generate_music_description_wav2clip(audio_path)
