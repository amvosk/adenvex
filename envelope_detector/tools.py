import torch
from torch.autograd import grad
from torch.utils.data import DataLoader
from .utils import SimpleDataset

def create_importance_indices(model, data, order=1, nbatch=100, device='cpu', grad_out=False):
    """
    Computes the importance indices of the model's output with respect to its intermediate outputs (`z`) based on gradient values.
    
    The function passes the input data through the model, computes gradients of the model's output (`y`) with respect to its
    intermediate outputs (`z`), and then aggregates these gradients to determine the importance of each element in `z`.
    
    Parameters:
    ----------
    model : torch.nn.Module
        The PyTorch model for which the importance indices need to be computed. The model is expected to return 
        intermediate outputs `z` and final outputs `y` when called.
        
    data : torch.Tensor or array-like
        Input data to be passed through the model. The shape is expected to be compatible with the model's input shape.
        
    order : int, optional (default=1)
        Order of the norm for gradient aggregation. 

    nbatch : int, optional (default=100)
        Batch size for processing the data in chunks.
        
    device : str, optional (default='cpu')
        The device to which the model and data will be moved before computation. Typically 'cuda' for GPU and 'cpu' for CPU.
        
    Returns:
    -------
    importance_indices : torch.Tensor
        Indices sorted by importance based on the aggregated gradient values. The index at the 0th position represents 
        the most important element in `z` and so on.
        
    gradients : torch.Tensor
        The aggregated gradients for all elements in `z`.
    """
    gradients = 0
    dataset = SimpleDataset(data)
    dataloader = DataLoader(dataset, batch_size=nbatch, shuffle=False)
    model = model.to(device)
    
    for batch in dataloader:
        batch = batch.to(device)
        z, y = model(batch)
        gradients_ = grad(outputs=y, inputs=z, grad_outputs=torch.ones_like(y), retain_graph=True)[0]
        gradients_norm = torch.sum(torch.abs(gradients_)**order, dim=(0,-1))**(1/order)
        gradients += gradients_norm
        
    importance_indices = torch.sort(gradients, descending=True).indices
    
    return importance_indices, gradients