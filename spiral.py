# spiral.py
# COMP9444, CSE, UNSW
# William Ye (z5061340)

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def to_polar(x, y):
  return (x**2 + y**2).sqrt(), torch.atan(y/x)

class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        self.in_to_hid = nn.Linear(2, num_hid)
        self.hid_to_out = nn.Linear(num_hid, 1)

    def forward(self, input):
        # Creating two separate tensors r and a for polar conversion
        r = (input[:,0] ** 2 + input[:,1] ** 2).sqrt()
        a = torch.atan2(input[:,1], input[:,0])
        # Stacking tensors r and a for feeding into nnetwork
        output = torch.stack((r, a), 1)
        output = self.in_to_hid(output)
        output = output.tanh()
        output = self.hid_to_out(output)
        output = torch.sigmoid(output)
        return output

class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        self.in_to_hid = nn.Linear(2, num_hid)
        self.hid_to_hid = nn.Linear(num_hid, num_hid)
        self.hid_to_out = nn.Linear(num_hid, 1)

    def forward(self, input):
        output = self.in_to_hid(input)
        self.hid1 = torch.tanh(output)
        output = self.hid_to_hid(self.hid1)
        self.hid2 = torch.tanh(output)
        output = self.hid_to_out(self.hid2)
        output = torch.sigmoid(output)
        return output

class ShortNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(ShortNet, self).__init__()
        self.in_to_hid1 = nn.Linear(2, num_hid)
        # Hidden Layer (HL)
        # Input into HL2 is of size HL1 + 2 (size of input)
        self.hid1_to_hid2 = nn.Linear(num_hid+2, num_hid)
        # Input into output is of size HL1 + HL2 + 2 (size of input)
        self.hid2_to_out = nn.Linear(num_hid+num_hid+2, 1)

    def forward(self, input):
        res1 = input
        output = self.in_to_hid1(input) 
        self.hid1 = torch.tanh(output)
        output = torch.cat((res1, self.hid1), 1) # Apply residual network
        output = self.hid1_to_hid2(output)
        self.hid2 = torch.tanh(output)
        res2 = self.hid2
        output = torch.cat((res1, res2, self.hid2), 1) # Apply residual network 
        output = self.hid2_to_out(output)
        output = torch.sigmoid(output)
        return output

def graph_hidden(net, layer, node):
    # xrange = torch.arange(start=-7,end=7.1,step=0.01,dtype=torch.float32)
    # yrange = torch.arange(start=-6.6,end=6.7,step=0.01,dtype=torch.float32)
    # xcoord = xrange.repeat(yrange.size()[0])
    # ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    # grid = torch.cat((xcoord.unsqueeze(1),ycoord.unsqueeze(1)),1)
    # print(grid.shape)

    # with torch.no_grad(): # suppress updating of gradients
    #     net.eval()        # toggle batch norm, dropout
    #     output = net.hid1
    #     net.train() # toggle batch norm, dropout back again
    #     print(output.shape)
    #     # predictions
    #     pred = (output >= 0.5).float()

    #     # plot function computed by model
    #     plt.clf()
    #     plt.pcolormesh(xrange,yrange,pred.cpu().view(yrange.size()[0],xrange.size()[0]), cmap='Wistia')
    return None
