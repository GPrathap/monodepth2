

#define policy model (model to learn a policy for my robot)
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 
from torch.autograd import Variable

class policy_gradient_model(nn.Module):
    def __init__(self):
        super(policy_gradient_model, self).__init__()
        self.fc0 = nn.Linear(2, 2)
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64,32)
        self.fc4 = nn.Linear(32,32)
        self.fc5 = nn.Linear(32, 2)
    def forward(self,x):
        self.output = []
        x1 = self.fc0(x)
        x2 = F.relu(self.fc1(x1))
        x3 = F.relu(self.fc2(x2))
        x4 = F.relu(self.fc3(x3))
        x5 = F.relu(self.fc4(x4))
        x6 = F.relu(self.fc5(x5))
        self.output.append(x6)
        self.output.append(x3)
        return self.output

policy_model = policy_gradient_model()
print(policy_model)
# optimizer = torch.optim.Adam(policy_model.parameters(), lr=0.005, betas=(0.9,0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)

inputs = torch.from_numpy(np.array([1, 2], dtype=np.float32)).requires_grad_()
model = policy_model(inputs)

model[0].backward(torch.Tensor([1, 2]))


print(inputs.grad)
