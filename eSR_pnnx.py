import os
import numpy as np
import tempfile, zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchvision
except:
    pass
# [1, 1, 486, 864]
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.filter = nn.Conv2d(bias=False, dilation=(1,1), groups=1, in_channels=1, kernel_size=(3,3), out_channels=12, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

        archive = zipfile.ZipFile('eSR.pnnx.bin', 'r')
        self.filter.weight = self.load_pnnx_bin_as_parameter(archive, 'filter.weight', (12,1,3,3), 'float32')
        archive.close()

    def load_pnnx_bin_as_parameter(self, archive, key, shape, dtype, requires_grad=True):
        return nn.Parameter(self.load_pnnx_bin_as_tensor(archive, key, shape, dtype), requires_grad)

    def load_pnnx_bin_as_tensor(self, archive, key, shape, dtype):
        fd, tmppath = tempfile.mkstemp()
        with os.fdopen(fd, 'wb') as tmpf, archive.open(key) as keyfile:
            tmpf.write(keyfile.read())
        m = np.memmap(tmppath, dtype=dtype, mode='r', shape=shape).copy()
        os.remove(tmppath)
        return torch.from_numpy(m)

    def forward(self, v_0):
        v_1 = self.filter(v_0)
        v_2 = self.pixel_shuffle(v_1)
        v_3, v_4 = torch.max(input=v_2, dim=1, keepdim=True)
        return v_3

def export_torchscript():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 1, 486, 864, dtype=torch.float)

    mod = torch.jit.trace(net, v_0)
    mod.save("eSR_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 1, 486, 864, dtype=torch.float)

    torch.onnx._export(net, v_0, "eSR_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0'], output_names=['out0'])

def test_inference():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 1, 486, 864, dtype=torch.float)

    return net(v_0)

if __name__ == "__main__":
    print(test_inference())
