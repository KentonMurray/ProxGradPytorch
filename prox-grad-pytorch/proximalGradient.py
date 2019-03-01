import torch
from torch.autograd import Variable
import torch.nn.functional as F

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

def linf1(parameter, bias=None, reg=0.01, lr=0.1):
    """Linfity1 Regularization using Proximal Gradients"""
#    print("parameter")
    
    #Norm = reg*lr
    Norm = 1.0 # TODO: Just for testing
    if bias is not None:
        w_and_b = torch.cat((parameter, bias.unfold(0,1,1)),1)
    else:
        w_and_b = parameter
    print("w_and_b:", w_and_b)
    sorted_w_and_b, indices = torch.sort(torch.abs(w_and_b), descending=True)
    desorted_w_and_b, indices = torch.sort(torch.abs(w_and_b), descending=False)
    print("sorted_w_and_b:", sorted_w_and_b)
    print("indices:", indices)
    #print(torch.einsum('ij->i', [sorted_w_and_b]))
    #print("topk:", torch.topk(sorted_w_and_b, 3))

    #SLOW
    rows, cols = sorted_w_and_b.size()
    print("rows, cols", rows, cols)

    sorted_z = torch.cat((sorted_w_and_b, torch.zeros(rows,1)),1)
    print("sorted_z:", sorted_z[:,1:])
    subtracted = sorted_w_and_b - sorted_z[:,1:]
    print("subtracted:", subtracted)

    scale_indices = torch.cumsum(torch.ones(rows,cols),1)
    print(scale_indices)
    scaled_subtracted = subtracted * scale_indices
    print(scaled_subtracted)
    max_mass = torch.cumsum(scaled_subtracted,1)
    print("max_mass:", max_mass)
    print(max_mass - Norm)
    nonzero = torch.clamp(-1*(max_mass - Norm),0)
    nonzero2 = torch.clamp((max_mass - Norm),0)
    print("NZ:", nonzero)

    oneN = 1.0/scale_indices


    nonzero_ones = torch.clamp(nonzero * 1000000, max=1)
    shifted_ones = torch.cat((torch.ones(rows,1),nonzero_ones[:,:(cols-1)]),1)
    over_one = -1*(nonzero_ones - shifted_ones)
    print("over_one:", over_one)
    last_one = torch.cat((over_one,torch.zeros(rows,1)),1)[:,1:]
    print("last_one:", last_one)
    max_remain = last_one * nonzero
    print("max_remain:", max_remain)
    shift_max = torch.cat((torch.zeros(rows,1),max_remain[:,:(cols-1)]),1)
    print("shift_max:", shift_max)
    tosub = nonzero_ones * subtracted + shift_max * oneN
    print("tosub:", tosub)

    nastyflipS = torch.flip(torch.flip(tosub,[0,1]),[0]) #TODO: Edge cases
    print(nastyflipS)
    aggsubS = torch.cumsum(nastyflipS,1)
    print(aggsubS)
    nastyflipagainS = torch.flip(torch.flip(aggsubS,[0,1]),[0]) #TODO: Edge cases
    print("NFAS:", nastyflipagainS)

    updated_weights = sorted_w_and_b - nastyflipagainS
    print("UPDATED WEIGHTS:", updated_weights)




    #still_need = Norm - torch.sum(nonzero,1)
    #still_need2 = Norm - torch.cumsum(nonzero,1)
    #print(still_need)
    #print(still_need2 * over_one)
    #updated_nonzero = nonzero + still_need2 * over_one
    #print("UNZ:", updated_nonzero)

    #print("NZ2:", nonzero2)
    #left = Norm - torch.cumsum(nonzero,1)
    #print("left:",left)
    ##maybe = torch.cat(((Norm - torch.cumsum(nonzero,1))[:,1:], torch.zeros(rows,1)),1)
    #maybe = torch.cat((torch.zeros(rows,1),left[:,:(cols-1)]),1)
    #print("maybe:",maybe)
    #print(torch.cumsum(maybe,1))
    #lm = left - maybe
    #print("l-m:", lm)
    #lm2 = lm * lm * 1000000 # Makes it positive
    #print("lm2:", lm2)
    #actual_ones = torch.clamp(lm2,max=1)
    #print(actual_ones)
    #actual_left = left * actual_ones
    #print(actual_left)
    #
    #
    #print(torch.cumsum(left-maybe,1))
    #print(Norm - torch.cumsum(nonzero[:,1:],1))
    #print(torch.cumsum(nonzero,1) - Norm)
    ##print(nonzero2)
    ##print(nonzero2 - torch.cumsum(nonzero,1))


    #print(oneN)
    #scaled_nonzero = oneN*nonzero
    #print("SN:", scaled_nonzero)
    #nastyflip = torch.flip(torch.flip(scaled_nonzero,[0,1]),[0]) #TODO: Edge cases
    #print(nastyflip)
    #aggsub = torch.cumsum(nastyflip,1)
    #print(aggsub)
    #nastyflipagain = torch.flip(torch.flip(aggsub,[0,1]),[0]) #TODO: Edge cases
    #print("NFA:", nastyflipagain)
    #print(Norm - torch.cumsum(nastyflipagain,1))
    ##print(torch.cumsum(scaled_nonzero,1))
    ##remainingMass = torch.clamp(Norm-torch.cumsum(nonzero,1),0)
    ##print("RM:",remainingMass)
    ##print(oneN*remainingMass)
    
    #print(torch.tensor([[-1, 1]]).size())
    #print(torch.nonzero(nonzero))
    #print(torch.pow((nonzero),0))
    #print(torch.unsqueeze(torch.nonzero(nonzero)[:,0], 0).size())
    #conv_mask = F.conv1d(torch.unsqueeze(torch.nonzero(nonzero)[:, 0], 0), torch.tensor([[-1, 1]]))
    #print("cm", conv_mask)
    #print(torch.nonzero(nonzero))
    #print(torch.unique(torch.nonzero(nonzero,)[:,0],return_inverse=True))
    #print(torch.nonzero(nonzero).sum(0))
    #print(torch.nonzero(nonzero).sum(1))
    #hacky = torch.cumsum(nonzero,1)
    #print("hacky", hacky)
    #print(torch.nonzero(torch.clamp(max_mass - Norm,0)))
    #print(torch.nonzero(torch.clamp(max_mass - Norm,0)))

    #TODO: val@index - (non-zero index - 1 / index)

    #remaining = torch.ones(rows)
    #remaining = remaining * Norm
    #for count in range(cols):
    #    if count == 0 :
    #        prev = sorted_w_and_b[:,count]
    #        continue
    #    current = sorted_w_and_b[:,count]
    #    print(prev, current)
    #    diff = (prev - current)*(count)
    #    print("diff:", diff)
    #    tosub = torch.min(remaining, diff)
    #    print("tosub:", tosub)
    #    #TODO:sub
    #    tosubi = tosub/count
    #    tosubm = torch.zeros(rows, cols)
    #    for newc in range(count):
    #        tosubm[:,newc] = tosubi
    #    #tosubm[:,:count-1] = tosubi
    #    print(tosubm)
    #    sorted_w_and_b = sorted_w_and_b - tosubm
    #    remaining = remaining - tosub
    #    print("remaining:", remaining)
    #
    #
    #    prev = current
    #    #TODO: Last column


    #for row in sorted_w_and_b:
    #    prev = -1.0
    #    total = 0.0
    #    count = -1
    #    newvals = zeros()#size of row
    #    for val in row:
    #        count = count + 1
    #        newvals[count] = 1.0
    #        if prev < 0.0:
    #            prev = val
    #            continue
    #        diff = prev - val
    #        if total + diff >= Norm:
    #            tosub = total + diff - Norm
    #            newval = prev - tosub/(1.0 + count)
    #            break
    #        else:
    #            total = total + diff
    #    recount = -1
    #    newvals = newvals * neval
    #    for val in row:
    #        recount = recount + 1
    #        if recount > count:
    #            break
    #        = val

    #cs = torch.cumsum(sorted_w_and_b, 1)
    #ds = torch.cumsum(desorted_w_and_b, 1)
    #print("cumsum:", cs)
    #print("decumsum:", ds)
    #print("cs-ds:", cs-ds)
    #print("subcumsum:", torch.cumsum(sorted_w_and_b, 1) - sorted_w_and_b)
    #tk, ti = torch.topk(w_and_b,3)
    #tks, tis = torch.sort(tk,descending=True)
    #tkcs = torch.cumsum(tks,1)
    #print("tkcs:", tkcs)
    #print("tkcs-tk:", tkcs - tks)
    #if w_and_b.is_cuda:
    #    ones_w = torch.ones(parameter.size(), device=torch.device("cuda"))
    #else:
    #    ones_w = torch.ones(parameter.size(), device=torch.device("cpu"))

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
#    print("parameter")
    
    if bias is not None:
        w_and_b = torch.cat((parameter, bias.unfold(0,1,1)),1)
    else:
        w_and_b = parameter
    #print("w_and_b:", w_and_b)
    L2 = reg # lambda: regularization strength
    Norm = (lr*L2/w_and_b.norm(2))
    if Norm.is_cuda:
        ones_w = torch.ones(parameter.size(), device=torch.device("cuda"))
    else:
        ones_w = torch.ones(parameter.size(), device=torch.device("cpu"))
    #print("Norm.device", Norm.device)
    #print("ones.device", ones.device)
    l2T = 1.0 - torch.min(ones_w, Norm)
    update = (parameter*l2T) #TODO: Figure out unsqueeze
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
    #print("pos:", pos)
    #print("neg:", neg)
    update_w = parameter - pos + neg
    #print("update_w:", update_w)
    parameter.data = update_w

    if bias is not None:
        if bias.is_cuda:
            Norms_b = Norm*torch.ones(bias.size(), device=torch.device("cuda"))
        else:
            Norms_b = Norm*torch.ones(bias.size(), device=torch.device("cpu"))
        pos = torch.min(Norms_b, Norm*torch.clamp(bias, min=0))
        neg = torch.min(Norms_b, -1.0*Norm*torch.clamp(bias, max=0))
        #print("pos:", pos)
        #print("neg:", neg)
        update_b = bias - pos + neg
        #print("update_b:", update_b)
        bias.data = update_b

def elasticnet(parameter, bias=None, reg=0.01, lr=0.1, gamma=1.0):
    """Elastic Net Regularization using Proximal Gradients.
    This is a linear combination of an l1 and a quadratic penalty."""
    if gamma < 0.0:
        print("Warning, gamma should be positive. Otherwise you are not shrinking.")
    #TODO: Is gamma of 1.0 a good value?
    Norm = reg*lr*gamma
    #print("parameter:", parameter)
    l1(parameter, bias, reg, lr)
    #print("parameter:", parameter)
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
    #print("squared:", squared)
    squared = squared + 4*Norm
    #print("squared:", squared)
    squareroot = torch.sqrt(squared)
    #print("squareroot:", squareroot)
    update_w = (parameter + squareroot)/2.0
    #print("parameter:", parameter)
    #print("update_w:", update_w)
    parameter.data = update_w

    if bias is not None:
        squared = torch.mul(bias, bias)
        #print("squared:", squared)
        squared = squared + 4*Norm
        #print("squared:", squared)
        squareroot = torch.sqrt(squared)
        #print("squareroot:", squareroot)
        update_b = (bias + squareroot)/2.0
        #print("bias:", bias)
        #print("update_b:", update_b)
        bias.data = update_b


