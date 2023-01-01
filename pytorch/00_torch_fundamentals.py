import torch
import numpy as np
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    print(torch.__version__)

    # scalar
    scalar = torch.tensor(7)
    print(scalar)
    print(scalar.ndim)
    print(scalar.item())

    # vector
    vector = torch.tensor([1, 2, 3])
    print(vector)
    print(vector.ndim)
    print(vector.shape)

    # matrix
    matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(matrix)
    print(matrix.ndim)
    print(matrix.shape)

    # tensor
    tensor = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    print(tensor)
    print(tensor.ndim)
    print(tensor.shape)
    print(tensor[0][0][0].item())

    # random tensor
    random_tensor = torch.rand(3, 4)
    print(random_tensor)
    print(random_tensor.ndim)

    # create random tensor to a similar shape of image
    random_image = torch.rand(size=(224, 224, 3))
    print(random_image.ndim)
    print(random_image.shape)

    # zeros and ones
    zeros = torch.zeros(3, 4)
    ones = torch.ones(3, 4)
    print(zeros)
    print(ones.dtype)

    # range of tensors
    range_tensor = torch.arange(3)
    print(range_tensor)
    print(range_tensor.ndim)
    print(range_tensor.shape)

    # create tensors like
    ten_zeros = torch.zeros_like(range_tensor)
    print(ten_zeros)
    print(ten_zeros.dtype)

    # ussual error but robustness is present
    ten_ones = torch.ones_like(range_tensor)
    float_32_tensor = torch.tensor([3, 6, 9])
    float_16_tensor = float_32_tensor.type(torch.float16)
    int_32_tensor = torch.tensor([3, 6, 9], dtype=torch.int32)
    print(float_16_tensor*float_32_tensor)
    print(int_32_tensor*float_32_tensor)

    # information about tensor
    some_tensor = torch.rand(3, 4)
    print(some_tensor)
    print(some_tensor.dtype)
    print(some_tensor.shape)
    print(some_tensor.device)

    # manipulating tensor
    tensor = torch.tensor([1, 2, 3])
    print(tensor + 100)
    print(tensor*10)
    print(torch.mul(tensor, 10))

    # matrix multiplication
    print(tensor*tensor)
    print(torch.matmul(tensor, tensor))

    # reshaping,stacking and others
    x = torch.arange(1, 13)
    print(x, x.shape)
    x_reshape = x.reshape(3, 4)
    print(x_reshape)

    # change view
    z = x.view(1, 12)
    print(z)
    print(z.shape)

    # tensor on cpu
    tensor = torch.tensor([1, 2, 3], device='cpu')
    tensor1 = torch.tensor([1, 2, 3])
    tensor2 = torch.tensor([1, 2, 3], device=device)
    print(tensor.device, tensor1.device, tensor2.device)
    tensor3 = tensor2.to('cpu')
    print(tensor3.device,tensor2.cpu().device)
    print(tensor2)
