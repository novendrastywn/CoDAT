import timm.utils
import torch
import onnxruntime as ort
import time
import timm
from timm import create_model
from model.build import *
from model.build import replace_batchnorm, reparameterize
from model.iformer import *
# from model import build
# import models.emo_model.emo
import utils
from fvcore.nn import FlopCountAnalysis, parameter_count
torch.autograd.set_grad_enabled(False)


T0 = 5
T1 = 10


def export_onnx(name, model, device, batch_size, resolution=224):
    model.to('cpu')
    model = model.eval()
    print("Convert To ONNX...")
    inputs = torch.randn(batch_size, 3, resolution, resolution, device='cpu')
    torch.onnx.export(model, inputs, f"./onnx_lat/{name}_{resolution}_{batch_size}.onnx", verbose = False, opset_version=12)
    inputs = torch.randn(1, 3, resolution, resolution, device='cpu')
    torch.onnx.export(model, inputs, f"./onnx_lat/{name}_{resolution}_{1}.onnx", verbose = False, opset_version=12)

    print("Finish...")

device = "cuda:0"

MODEL_LIST=[
    ('parformer_s1', 256),
    ('parformer_s2', 256),
    ('parformer_s3', 256),
    ('shvit_s3', 256),
    ('shvit_s4', 256),
    ('shvit_s4', 384),
    ('deit_base_patch16', 224),

]

from argparse import ArgumentParser
import torchvision

parser = ArgumentParser()

parser.add_argument('--model', default='repinc_m2_3', type=str) #repinc_m1
parser.add_argument('--resolution', default=256, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--checkpoint', default=None)

if __name__ == "__main__":
    args = parser.parse_args()
    # model_name = args.model
    # batch_size = args.batch_size
    # resolution = args.resolution
    for model_name, resolution in MODEL_LIST:
        torch.cuda.empty_cache()
        if model_name == 'shufflenet_v2_x1_0':
            model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
        elif model_name == 'shufflenet_v2_x1_5':
            model = torchvision.models.shufflenet_v2_x1_5(pretrained=True)
        elif model_name == 'shufflenet_v2_x2_0':
            model = torchvision.models.shufflenet_v2_x2_0(pretrained=True)
            # model = torchvision.models.mobilenet
        else:
            model = create_model(model_name, num_classes=1000) #inference_mode=True,

        model=timm.utils.reparameterize_model(model)
        model = reparameterize(model.eval())
        # torch.onnx.export(model, inputs, './onnx/'+args.model+".onnx")
        from ptflops import get_model_complexity_info
        macs, n_parameters = get_model_complexity_info(
                            model, (3, resolution, resolution), as_strings=False,
                            print_per_layer_stat=False, verbose=False)
        gmacs = macs / (1000**3)
        print(f"{model_name}, params: {(n_parameters/(1000**2)):.2f} M, macs: {gmacs:.3f} G")
        # inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)
        export_onnx(model_name, model, device='cpu', batch_size=args.batch_size, resolution=resolution)
