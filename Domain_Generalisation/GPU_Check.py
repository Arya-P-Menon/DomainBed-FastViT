# python-3.10 -m venv spsd
# source spsd/bin/activate
# pip install -r requirements.txt
# pip uninstall torch torchvision torchaudio
# pip cache purge
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

import torch
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
