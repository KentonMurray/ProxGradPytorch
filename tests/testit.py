import torch
import proximal_gradient.proximalGradient as pg

class OneLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        The constructor creates one linear layer and assigns it a name.
        """
        super(OneLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, D_out)
        self.linear1.name = "linear1"

    def forward(self, x):
        """
        Simple forward step
        """
        y_pred = self.linear1(x)
        # Uncomment for verbose debugging
        #print("linear1:", self.linear1)
        #for param in self.linear1.parameters():
        #    print("param:", param)
        #    print("param.grad:", param.grad)
        ##print("linear1.grad:", self.linear1.grad)
        ##print("linear1.grad:", self.linear1.data)
        return y_pred

def build_model():
    # Values for the network size
    N, D_in, H, D_out = 4, 3, 4, 2
    #N, D_in, H, D_out = 4, 3, 10, 5

    # Create random Tensors to hold inputs and outputs
    x = torch.zeros(N, D_in)
    y = torch.ones(N, D_out)
    print("x.requires_grad")
    print(x.requires_grad)

    # Construct our model by instantiating the class defined above
    model = OneLayerNet(D_in, H, D_out)
    print("model:", model)

    criterion = torch.nn.MSELoss(size_average=False)
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    return (x,y,model,criterion,optimizer)


def test_l1(network):
    x, y, model, criterion, optimizer = network
    for t in range(10):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        cross_entropy_loss = criterion(y_pred, y)
        loss = cross_entropy_loss

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
#        print("model.linear1.weight.grad:", model.linear1.weight.grad)
#        print("model.linear1.bias.grad:", model.linear1.bias.grad)
#        print("model.linear1.weight before:", model.linear1.weight)
#        print("model.linear1.bias before:", model.linear1.bias)
        optimizer.step()
#        print("model.linear1.weight after:", model.linear1.weight)
#        print("model.linear1.bias after:", model.linear1.bias)
#        print("model.linear1.weight.norm():", model.linear1.weight.norm())

        #L1...
        print("weight before:", model.linear1.weight)
        pg.l1(model.linear1.weight, model.linear1.bias, reg=0.1)
        print("weight after:", model.linear1.weight)

def test_l21(network):
    print("Testing l21")
    x, y, model, criterion, optimizer = network
    for t in range(10):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        cross_entropy_loss = criterion(y_pred, y)
        loss = cross_entropy_loss

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #print("weight before:", model.linear1.weight)
        #print("bias before:", model.linear1.bias)
        #pg.l21(model.linear1.weight, reg=0.1) # Use defaults
        pg.l21(model.linear1.weight, model.linear1.bias, reg=0.1)
        #pg.l21(model.linear1, reg=0.01) # Test different learning rates
        #pg.l21(model.linear1, reg=0.1)
        #pg.l21_slow(model.linear1.weight, reg=0.1) # Slow version to double check accuracy
        #print("weight after:", model.linear1.weight)
        #print("bias after:", model.linear1.bias)

def test_l2(network):
    print("Testing l2")
    x, y, model, criterion, optimizer = network
    for t in range(10):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        cross_entropy_loss = criterion(y_pred, y)
        loss = cross_entropy_loss

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Seeing about l2...
        #print("L2:", model.linear1.weight.norm(2))
        #print("weight before:", model.linear1.weight)
        pg.l2(model.linear1.weight, model.linear1.bias, reg=0.1)
        #print("weight after:", model.linear1.weight)


def test_linf1(network):
    x, y, model, criterion, optimizer = network
    for t in range(500):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        cross_entropy_loss = criterion(y_pred, y)
        loss = cross_entropy_loss

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #Linf1...
        print("weight before:", model.linear1.weight)
        print("bias before:", model.linear1.bias)
        pg.linf1(model.linear1.weight, model.linear1.bias, reg=0.1)
        print("weight after:", model.linear1.weight)
        print("bias after:", model.linear1.bias)


def test_linf(network):
    x, y, model, criterion, optimizer = network
    for t in range(10):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        cross_entropy_loss = criterion(y_pred, y)
        loss = cross_entropy_loss

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #Linf
        print("weight before:", model.linear1.weight)
        pg.linf(model.linear1.weight, model.linear1.bias, reg=0.1)
        print("weight after:", model.linear1.weight)
   

def test_elasticnet(network):
    print("Testing elasticnet")
    x, y, model, criterion, optimizer = network
    for t in range(10):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        cross_entropy_loss = criterion(y_pred, y)
        loss = cross_entropy_loss

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        #Elastic Net
        #print("weight before:", model.linear1.weight)
        pg.elasticnet(model.linear1.weight, model.linear1.bias, reg=0.1)
        #print("weight after:", model.linear1.weight)
 
def test_logbarrier(network):
    print("Testing logbarrier")
    x, y, model, criterion, optimizer = network
    for t in range(10):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        cross_entropy_loss = criterion(y_pred, y)
        loss = cross_entropy_loss

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #Log Barrier...
        #print("weight before:", model.linear1.weight)
        pg.logbarrier(model.linear1.weight, model.linear1.bias, reg=0.1)
        #print("weight after:", model.linear1.weight)


def main():
    network = build_model()
    test_l1(network)    
    test_linf1(network) 
    test_elasticnet(network)    
    test_logbarrier(network)
    test_l2(network)    
    test_l21(network)    

if __name__ == "__main__":
    main()
