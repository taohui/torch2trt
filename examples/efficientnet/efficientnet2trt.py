import sys
import time
import numpy as np
import torch
from torch2trt import torch2trt
from torchvision.models.alexnet import alexnet
from efficient_and_vggish_embedding.rgb_audio_feature_extraction import RgbAudioExtractorClass
from torch2trt import TRTModule

model_dir = "/dockerdata/vggmodels/"

MINI_EXTRACTOR = RgbAudioExtractorClass(model_path=model_dir, is_xiaoshipin=True)
model = MINI_EXTRACTOR.rgb_model.module

root = "/dockerdata/data/"
video_path = root + "13841411589535373331-0_video_is_min.npy"
x = torch.from_numpy(np.load(video_path))[0:1].cuda()
print(x.shape)

# create some regular pytorch model...
#model = alexnet(pretrained=True).eval().cuda()

# create example data
#x = torch.ones((1, 3, 300, 300)).cuda()

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [x], max_batch_size=64, fp16_mode=True, max_workspace_size=1<<31, use_onnx=True, dynamic_axes={"input_0":[0], "output_0":[0]}, dynamic_shapes={"input_0":([1, 3, 300, 300], [32, 3, 300, 300], [64, 3, 300, 300])})
torch.save(model_trt.state_dict(), 'efficientnet_trt_onnx.pth')

model_trt = TRTModule()

model_trt.load_state_dict(torch.load('efficientnet_trt_onnx.pth'))

a = time.time()
y = model(x)
print(y)
print(time.time()-a)
for i in range(100):
    b = time.time()
    y_trt = model_trt(x)
    print(time.time()-b)
print(y_trt)
# check the output against PyTorch
print(torch.sum(torch.abs(y)), torch.sum(torch.abs(y_trt)))
print(torch.max(torch.abs(y - y_trt)))
