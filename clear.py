# import tensorflow as tf
import torch

def clear_gpu_memory():
    # Clear TensorFlow GPU memory
    # tf.keras.backend.clear_session()
    # tf.compat.v1.reset_default_graph()

    # Clear PyTorch GPU memory
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    print("GPU memory cleared.")

# Execute the function to clear GPU memory
clear_gpu_memory()
