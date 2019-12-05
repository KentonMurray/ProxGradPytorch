import torch
from torch.autograd import Variable
import torch.nn.functional as F

def package():
    print("Kenton")

def l21(parameter, bias=None, reg=0.01, lr=0.1):
    """L21 Regularization"""
    
    if bias is not None:
        w_and_b = torch.cat((parameter, bias.unfold(0,1,1)),1)
    else:
        w_and_b = parameter
    L21 = reg # lambda: regularization strength
    Norm = (lr*L21/w_and_b.norm(2, dim=1))
    if Norm.is_cuda:
        ones = torch.ones(w_and_b.size(0), device=torch.device("cuda"))
    else:
        ones = torch.ones(w_and_b.size(0), device=torch.device("cpu"))
    l21T = 1.0 - torch.min(ones, Norm)
    update = (parameter*(l21T.unsqueeze(1)))
    parameter.data = update
    # Update bias
    if bias is not None:
        update_b = (bias*l21T)
        bias.data = update_b

def l21_slow(parameter, reg=0.01, lr=0.1):
    """L21 Regularization (Slow implementation. Used for
       sanity checks.)"""
    
    w_and_b = parameter
    l21s = []
    for row in w_and_b:
        L21 = reg
        l21 = lr * L21/row.norm(2) 
        l21 = 1.0 - min(1.0, l21) 
        l21s.append(l21)
    counter = 0
    for row in parameter:
        updated = row * l21s[counter]
        parameter.data[counter] = updated
        counter = counter + 1
    counter = 0

def linf1(parameter, bias=None, reg=0.01, lr=0.1):
    """Linfity1 Regularization using Proximal Gradients"""

    Norm = reg*lr
    if bias is not None:
        w_and_b = torch.cat((parameter, bias.unfold(0,1,1)),1)
    else:
        w_and_b = parameter
    sorted_w_and_b, indices = torch.sort(torch.abs(w_and_b), descending=True)

    # CUDA or CPU
    devicetype="cuda"
    if w_and_b.is_cuda:
        devicetype="cuda"
    else:
        devicetype="cpu"


    #SLOW
    rows, cols = sorted_w_and_b.size()

    sorted_z = torch.cat((sorted_w_and_b, torch.zeros(rows,1, device=torch.device(devicetype))),1)
    subtracted = torch.clamp(sorted_w_and_b - sorted_z[:,1:],max=Norm) #Max=Norm important

    scale_indices = torch.cumsum(torch.ones(rows,cols, device=torch.device(devicetype)),1)
    scaled_subtracted = subtracted * scale_indices
    max_mass = torch.cumsum(scaled_subtracted,1)
    nonzero = torch.clamp(-1*(max_mass - Norm),0)

    oneN = 1.0/scale_indices


    nonzero_ones = torch.clamp(nonzero * 1000000, max=1) #TODO: Hacky
    shifted_ones = torch.cat((torch.ones(rows,1, device=torch.device(devicetype)),nonzero_ones[:,:(cols-1)]),1)
    over_one = -1*(nonzero_ones - shifted_ones)
    last_one = torch.cat((over_one,torch.zeros(rows,1, device=torch.device(devicetype))),1)[:,1:]
    max_remain = last_one * nonzero
    shift_max = torch.cat((torch.zeros(rows,1, device=torch.device(devicetype)),max_remain[:,:(cols-1)]),1)
    first_col_nonzero_ones = torch.cat((torch.ones(rows,1, device=torch.device(devicetype)),nonzero_ones[:,1:]),1) #Edge case for only first column
    tosub = first_col_nonzero_ones * subtracted + shift_max * oneN

    nastyflipS = torch.flip(torch.flip(tosub,[0,1]),[0]) #TODO: Edge cases
    aggsubS = torch.cumsum(nastyflipS,1)
    nastyflipagainS = torch.flip(torch.flip(aggsubS,[0,1]),[0]) #TODO: Edge cases

    updated_weights = sorted_w_and_b - nastyflipagainS
    unsorted = torch.zeros(rows,cols, device=torch.device(devicetype)).scatter_(1,indices,updated_weights)
    final_w_and_b = torch.sign(w_and_b) * unsorted

    # Actually update parameters and bias
    if bias is not None:
        update = final_w_and_b[:,:cols-1]
        parameter.data = update
        update_b = final_w_and_b[:,-1]
        bias.data = update_b
    else:
        parameter.data = final_w_and_b
        



def linf(parameter, bias=None, reg=0.01, lr=0.1):
    """L Infinity Regularization using proximal gradients over entire tensor"""
    
    if bias is not None:
        w_and_b = torch.squeeze(torch.cat((parameter, bias.unfold(0,1,1)),1), 0)
    else:
        w_and_b = torch.squeeze(parameter, 0)
    print("w_and_b:", w_and_b)
    sorted_w_and_b, indices = torch.sort(torch.abs(w_and_b), descending=True)
    print("sorted_w_and_b:", sorted_w_and_b)
    


def l2(parameter, bias=None, reg=0.01, lr=0.1):
    """L2 Regularization over the entire parameter's values using proximal gradients"""
    
    if bias is not None:
        w_and_b = torch.cat((parameter, bias.unfold(0,1,1)),1)
    else:
        w_and_b = parameter
    L2 = reg # lambda: regularization strength
    Norm = (lr*L2/w_and_b.norm(2))
    if Norm.is_cuda:
        ones_w = torch.ones(parameter.size(), device=torch.device("cuda"))
    else:
        ones_w = torch.ones(parameter.size(), device=torch.device("cpu"))
    l2T = 1.0 - torch.min(ones_w, Norm)
    update = (parameter*l2T) 
    parameter.data = update
    # Update bias
    if bias is not None:
        if Norm.is_cuda:
            ones_b = torch.ones(bias.size(), device=torch.device("cuda"))
        else:
            ones_b = torch.ones(bias.size(), device=torch.device("cpu"))
        l2T = 1.0 - torch.min(ones_b, bias)
        update_b = (bias*l2T)
        bias.data = update_b

def l1(parameter, bias=None, reg=0.01, lr=0.1):
    """L1 Regularization using Proximal Gradients"""
    Norm = reg*lr

    # Update W
    if parameter.is_cuda:
        Norms_w = Norm*torch.ones(parameter.size(), device=torch.device("cuda"))
    else:
        Norms_w = Norm*torch.ones(parameter.size(), device=torch.device("cpu"))
    pos = torch.min(Norms_w, Norm*torch.clamp(parameter, min=0))
    neg = torch.min(Norms_w, -1.0*Norm*torch.clamp(parameter, max=0))
    update_w = parameter - pos + neg
    parameter.data = update_w

    if bias is not None:
        if bias.is_cuda:
            Norms_b = Norm*torch.ones(bias.size(), device=torch.device("cuda"))
        else:
            Norms_b = Norm*torch.ones(bias.size(), device=torch.device("cpu"))
        pos = torch.min(Norms_b, Norm*torch.clamp(bias, min=0))
        neg = torch.min(Norms_b, -1.0*Norm*torch.clamp(bias, max=0))
        update_b = bias - pos + neg
        bias.data = update_b

def elasticnet(parameter, bias=None, reg=0.01, lr=0.1, gamma=1.0):
    """Elastic Net Regularization using Proximal Gradients.
    This is a linear combination of an l1 and a quadratic penalty."""
    if gamma < 0.0:
        print("Warning, gamma should be positive. Otherwise you are not shrinking.")
    #TODO: Is gamma of 1.0 a good value?
    Norm = reg*lr*gamma
    l1(parameter, bias, reg, lr)
    update_w = (1.0/(1.0 + Norm))*parameter
    parameter.data = update_w
    if bias is not None:
        update_b = (1.0/(1.0 + Norm))*bias
        bias.data = update_b

def logbarrier(parameter, bias=None, reg=0.01, lr=0.1):
    """Project onto logbarrier. Useful for minimization
    of f(x) when x >= b.
    F(A) = -log(det(A))"""
    #TODO: Verify this is correct
    Norm = reg*lr

    # Update W
    squared = torch.mul(parameter, parameter)
    squared = squared + 4*Norm
    squareroot = torch.sqrt(squared)
    update_w = (parameter + squareroot)/2.0
    parameter.data = update_w

    if bias is not None:
        squared = torch.mul(bias, bias)
        squared = squared + 4*Norm
        squareroot = torch.sqrt(squared)
        update_b = (bias + squareroot)/2.0
        bias.data = update_b


