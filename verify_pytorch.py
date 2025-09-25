import torch

if torch.cuda.is_available():
    print("CUDA is available. PyTorch can utilize the GPU.")
    print("CUDA version: {}".format(torch.version.cuda))
    print("pyTorch version:{}".format(torch.__version__))
    print("GPU device name: {}".format(torch.cuda.get_device_name(0)))
    print("CuDNN version: {}".format(torch.backends.cudnn.version()))
    print("number of CUDA devices: {}".format(torch.cuda.device_count()))
    print("Current CUDA device index: {}".format(torch.cuda.current_device()))
    print("Device name: {}".format(torch.cuda.get_device_name(torch.cuda.current_device())))
else:
    print("CUDA is not available. PyTorch will use the CPU.")    