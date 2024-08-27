import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import torchvision.transforms.functional as TF

from models import edgeSR_MAX, edgeSR_TM, edgeSR_CNN, edgeSR_TR, FSRCNN, ESPCN, Classic
import torch.onnx

device = torch.device('cpu')
model_file = 'model-files/eSR-MAX_s2_K3_C14.model'
model_id = model_file.split('.')[-2].split('/')[-1]

# 加载模型
if model_id.startswith('eSR-MAX_'):
    model = edgeSR_MAX(model_id)
elif model_id.startswith('eSR-TM_'):
    model = edgeSR_TM(model_id)
elif model_id.startswith('eSR-TR_'):
    model = edgeSR_TR(model_id)

elif model_id.startswith('FSRCNN_'):
    model = FSRCNN(model_id)
elif model_id.startswith('ESPCN_'):
    model = ESPCN(model_id)
elif model_id.startswith('Bicubic_'):
    model = Classic(model_id)
else:
    assert False

model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage), strict=True)
model.to(device).eval()

# 导出为ONNX格式
dummy_input = torch.randn(1, 1, 256, 256, device=device)  # 根据实际输入尺寸修改
torch.onnx.export(model, dummy_input, "model.onnx", verbose=True)