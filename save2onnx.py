from typing import Optional, Generator
import os
import math
import random
import itertools
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import torchinfo
import whisper
import IPython.display as ipd
from torch.utils.mobile_optimizer import optimize_for_mobile

from transformers import GPT2TokenizerFast
import inspect


whisper_path = Path("~/.cache/whisper/tiny.en.pt").expanduser()  # download the weights first using official whisper
with open(whisper_path, "rb") as f:
    checkpoint = torch.load(f)
ipd.display(checkpoint.keys())
ipd.display(checkpoint["dims"])

# This is to get mel shape
# audiofile = "outputfile.mp3"
# audio = whisper.load_audio(audiofile)
# audio = whisper.pad_or_trim(audio)
# mel = whisper.log_mel_spectrogram(audio).unsqueeze(0)
# print(mel.size())   # confirmed torch.Size([1, 80, 3000])

ori = whisper.load_model("tiny", device="cpu").eval()  # original model loading
# scripted_encoder = torch.jit.script(ori.encoder).eval()

# Get the names of the input arguments from the forward method
# Input names is odict_keys(['x'])
# input_names = inspect.signature(ori.encoder.forward).parameters.keys()
# print("Input names:", input_names)

# torch_input = torch.randn(1, 8, 3000)
# onnx_encoder = torch.onnx.dynamo_export(ori.encoder, torch_input)
# onnx_encoder.save("whisper_ori_encoder.onnx")

# encoder model - export 
torch_input = torch.randn(1, 80, 3000)
onnx_program = torch.onnx.export(ori.encoder, 
                                torch_input, 
                                "whisper_ori_encoder.onnx",
                                export_params=True,
                                opset_version=10,
                                do_constant_folding=True,
                                input_names = ['ori.encoder.forward.x'])


# decoder model - export 
tensor_size = (1, 448)
torch_input_x = torch.randint(high=100, size=tensor_size, dtype=torch.long)
# torch_input_x = torch.randn(1, 448, dtype=torch.long) 
torch_input_xa = torch.randn(1, 1500, 384)
onnx_program = torch.onnx.export(ori.decoder, 
                                (torch_input_x, torch_input_xa), 
                                "whisper_ori_decoder.onnx",
                                export_params=True,
                                opset_version=10,
                                do_constant_folding=True,
                                input_names = ['ori.decoder.forward.x', 'ori.decoder.forward.xa'])