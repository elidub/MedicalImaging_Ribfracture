import torch

def crop(x, l = -1):
    x = x.unsqueeze(1) # Unsqueeze the channel dimension
    assert len(x.shape) == 5, "Input tensor must have 5 dimensions"
    if l != -1: x = x[:, :, :l, :l, :l] # Crop the input tensor if l is specified
    return x

def label_processor(self, y):
    # This is a placeholder function, such that it works with the dummy network!!!
    y = y.float()
    y = y[:, :, :, 0]  # Only select the first slice, similar to the dummy network
    y_downsampled = torch.nn.functional.avg_pool2d(y, kernel_size = 2*7, stride = 2**7).view(-1, 16)
    return y_downsampled