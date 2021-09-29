from skimage import data
import matplotlib.pyplot as plt
import torch

cat = data.chelsea()

torch_cat = torch.tensor(cat, dtype=torch.float32)
