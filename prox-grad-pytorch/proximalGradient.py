import torch
from torch.autograd import Variable

def package():
    print("Kenton")

def l21_slow(parameter, reg=0.01, lr=0.1):
    """L21 Regularization"""
#    print("parameter")
    
#    w_and_b = torch.cat((parameter.weight, parameter.bias.unfold(0,1,1)),1) #TODO:
    w_and_b = parameter
    #print("w_and_b:", w_and_b)
    #for x in range(parameter.weight):
    #  print("x:", x)
    l21s = []
    for row in w_and_b:
#        print("row:", row)
#        print("row.norm(2):", row.norm(2))
        #L21 = 0.1
        L21 = reg
#        print(reg, lr)
        l21 = lr * L21/row.norm(2) #TODO
#        l21 = 0.21 * L21/row.norm(2)
        l21 = 1.0 - min(1.0, l21) #TODO
#        print("l21:", l21)
        l21s.append(l21)
    #counter = 0
    #for row in parameter.weight:
    #  print("row:", row)
    #  print("parameter.bias.expand(1):", parameter.bias.expand(0))
    #  #print("torch.cat(row, parameter.bias[counter]):", torch.cat((row, parameter.bias[counter])))
    #  print("row.norm(2):", row.norm(2))
    #  print("parameter.bias[counter].norm(2):", parameter.bias[counter].norm(2))
    #  counter = counter + 1
#    print("parameter.bias.norm():", parameter.bias.norm())
    counter = 0
#    for row in parameter.weight:
    for row in parameter:
#        print("row:", row)
#        print("l21s[counter]", l21s[counter])
        updated = row * l21s[counter]
#        print("updated:", updated)
        #row = row * l21s[counter]
#        parameter.weight.data[counter] = updated
        parameter.data[counter] = updated
        counter = counter + 1
#    print(parameter.weight)
#    print(parameter)
    counter = 0
#    for row in parameter.bias:
#        print("row:", row)
#        print("l21s[counter]", l21s[counter])
#        updated = row * l21s[counter]
#        print("updated:", updated)
#        #row = row * l21s[counter]
#        parameter.bias.data[counter] = updated
#        counter = counter + 1
#    print(parameter.bias)

#    print("parameter.weight")
#    print(parameter.weight)
#    print(parameter.bias)

def l21(parameter, bias=None, reg=0.01, lr=0.1):
    """L21 Regularization"""
#    print("parameter")
    
#    w_and_b = torch.cat((parameter.weight, parameter.bias.unfold(0,1,1)),1) #TODO:
    if bias is not None:
        w_and_b = torch.cat((parameter, bias.unfold(0,1,1)),1)
    else:
        w_and_b = parameter
    #print("w_and_b:", w_and_b)
    L21 = reg # lambda: regularization strength
    Norm = (lr*L21/w_and_b.norm(2, dim=1))
    if Norm.is_cuda:
        ones = torch.ones(w_and_b.size(0), device=torch.device("cuda"))
    else:
        ones = torch.ones(w_and_b.size(0), device=torch.device("cpu"))
    #print(ones)
    #print(Norm)
    #print("Norm.device", Norm.device)
    #print("ones.device", ones.device)
    l21T = 1.0 - torch.min(ones, Norm)
    #print("Tensor:", l21T)
    update = (parameter*(l21T.unsqueeze(1)))
    #print("MULT:", update)
    #print("param1:", parameter)
    parameter.data = update
    #print("param2:", parameter)
    # Update bias
    if bias is not None:
        #print("bias1:", bias)
        #print(l21T)
        #update_b = (bias*(l21T.unsqueeze(1)))
        update_b = (bias*l21T)
        bias.data = update_b
        #print("bias2:", bias)

