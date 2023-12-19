import torch
import torch.nn as nn
import numpy as np
def Improve1_RobustCrossEntropyLoss(x=torch.rand((2,2,48,224,224)),y=torch.rand((2,48,224,224)),weight=torch.FloatTensor([1, 3]).to(torch.device('cuda:0'))):
      softmax=nn.Softmax(dim=1)
      x_softmax=softmax(x)
      # w1=x_softmax[y]
      x_log=torch.log(x_softmax+1e-15)
      # print([range(len(x)), y])
      # y=y.view(-1, 1)
      # loss = x_log[y]
      # mask1 = np.copy(y)
      # mask1=np.copy(y)
      # mask1=torch.rand((2,1,48,224,224)).to(torch.device('cuda:0'))
      # mask2=torch.rand((2,1,48,224,224)).to(torch.device('cuda:0'))
      mask1 = torch.rand(y.shape).to(torch.device('cuda:0'))
      mask2 = torch.rand(y.shape).to(torch.device('cuda:0'))
      mask2=mask2.bool()
      # mask1.copy_(y)
      y = y.long()
      temp3 = torch.gather(x_log, dim=1, index=y)
      temp4 = -temp3
      temp5= torch.gather(x_softmax, dim=1, index=y)
      # mask1[mask1==0]=weight[0]
      # mask1[mask1==1]=weight[1]
      mask1[y==0]=weight[0]
      mask1[y==1]=weight[1]
      x_w=mask1*temp4
      mask2[y==0]=temp5[y==0]>0.1
      mask2[y == 1] = temp5[y == 1] <0.9
      x_w_i=mask2*x_w
      # temp_sum=x_w.sum()
      # temp_b = temp4[y == 0,:]
      # temp_b=temp4[y==0]*weight[0]
      # temp_v=temp4[y==1]*weight[1]
      # temp_sum=temp_b.sum()+temp_v.sum()

      # y = y[:, 0].long()
      # loss_func = nn.NLLLoss(reduction='none')
      # loss=loss_func(x_log, y)



      # loss = x_log[range(len(x)),y]
      return x_w_i,mask2


