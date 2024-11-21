# Generate music style prompts from the music files
from __future__ import annotations
import cv2
import copy
import torch
import einops
import mediapy
import numpy as np
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
from typing import Any
from typing import Callable
from dataclasses import dataclass
from diffusers.utils import load_image
from torch.nn import functional as nnf
from diffusers.models import attention_processor
from diffusers.image_processor import PipelineImageInput
from transformers import DPTImageProcessor, DPTForDepthEstimation
from diffusers.utils.torch_utils import is_compiled_module, is_torch_version
from diffusers import StableDiffusionXLPipeline, DDIMScheduler, ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL

from src.Handler import Handler
from src.StyleAlignedArgs import StyleAlignedArgs
from src.Tokenization_and_Embedding import prompt_tokenizazion_and_embedding, embeddings_ensemble, embeddings_ensemble_with_neg_conditioning
from src.Encode_Image import image_encoding
from src.Diffusion import Generate_Noise_Prediction, Denoising_next_step, DDIM_Process, extract_latent_and_inversion, DDIM_Inversion_Process


# For the Music Model (Content AudioMusic).
import hashlib
import torchaudio
# from laion_clap import CLAP_Module

# For the Summarization and rephrasing.
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Create Alias for torch.tensor to increase readability.
T = torch.tensor
TN = T
import hashlib
from laion_clap import CLAP_Module
import os
import glob

# Function to generate a unique seed from the audio file
def generate_seed_from_audio(audio_path):
    with open(audio_path, 'rb') as f:
        audio_data = f.read()
    return int(hashlib.md5(audio_data).hexdigest(), 16) % (2**32)

# Function to generate music description based on tonal characteristics
def generate_music_description(audio_path):
    # Generate a unique seed based on the audio file
    seed = generate_seed_from_audio(audio_path)

    # Set the seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Instantiate the CLAP model
    model = CLAP_Module()

    # Load your audio file
    waveform, sample_rate = torchaudio.load(audio_path)

    # Resample if necessary (CLAP expects 48kHz)
    if sample_rate != 48000:
        resampler = torchaudio.transforms.Resample(sample_rate, 48000)
        waveform = resampler(waveform)

    # Make sure it's mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Ensure the tensor is on the CPU
    audio_tensor = waveform.clone().detach().cpu()

    # Tonal characteristic queries (atmospheric and descriptive)
    tonal_queries = [
        # Moods and Atmosphere
        # "Little birds singing in the trees", "Wolves at scary forest",
        # "Dog barking twice", "Sweet kitty meow",
        # "Rooster crowing in the morning", "Cartoon kitty begging meow",
        # "Wolf howling", "Aggressive beast roar"

        "Cat", "Motorcycle","Wolf", "Ambulance"
    ]

    # Function to get similarities
    def get_similarities(audio_tensor, queries):
        # Ensure audio tensor is on the CPU
        audio_embeddings = torch.tensor(model.get_audio_embedding_from_data(x=audio_tensor.numpy()))
        # Get text embeddings (which are already on the CPU as a NumPy array)
        text_embeddings = model.get_text_embedding(queries)

        # Calculate similarities
        similarities = (audio_embeddings @ torch.tensor(text_embeddings).T).squeeze()

        # # Print the similarities for debugging
        # for query, similarity in zip(queries, similarities):
        #     print(f"{query} / {similarity.item()}")
        
        result = sorted(zip(queries, similarities), key=lambda x: x[1], reverse=True)
        return result

    # Get tonal similarities
    tonal_results = get_similarities(audio_tensor, tonal_queries)
    # Select the top tonal description
    top_tonal_description = tonal_results[0][0]

    # Generate the final tonal description
    # combined_description = f"The atmosphere is {top_tonal_description}."
    combined_description = f"{top_tonal_description}."

    # print("Generated Music Description:")
    print(combined_description+"\n")

    return combined_description


PATH = "asset/audio_animal/mixkit-ambulance-siren-us-1642.wav"
music_style_prompt = generate_music_description(PATH)
# Path to the audio animal folder


# audio_folder_path = "asset/audio_animal/"

# # Get all audio files in the folder
# audio_files = glob.glob(os.path.join(audio_folder_path, "*.wav"))

# # Iterate over each audio file and generate music descriptions
# for audio_file in audio_files:
#     print(f"[PROCESSING] {audio_file}")
#     music_style_prompt = generate_music_description(audio_file)
#     print("Music Style Caption:", music_style_prompt)