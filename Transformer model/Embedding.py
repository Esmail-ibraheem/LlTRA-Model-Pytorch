import torch 
import torch.nn as nn 
from torch.optim import Adam 
from torch.distributions.uniform import Uniform
from torch.utils.data import TensorDataset, DataLoader

import lightning as L 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

inputs = torch.tensor([[1, 0, 0, 0], 
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

labels = torch.tensor([[0, 1, 0, 0], 
                       [0, 0, 1, 0],
                       [0, 0, 0, 1],
                       [0, 1, 0, 0]])

dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)

class WordEmbeddingFromScratch(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        min_value = -0.5 
        max_value = 0.5 

        self.input1_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input1_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input2_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input2_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input3_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input3_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input4_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input4_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        
        self.output1_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output1_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output2_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output2_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output3_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output3_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output4_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output4_w2 = nn.Parameter(Uniform(min_value, max_value).sample())

        self.loss_Function = nn.CrossEntropyLoss()
    def forward(self, input):
        input = input[0]

        inputs_to_top_hidden = ((input[0] * self.input1_w1) + 
                                (input[1] * self.input2_w1) + 
                                (input[2] * self.input3_w1) + 
                                (input[3] * self.input4_w1))
        
        
        inputs_to_bottom_hidden = ((input[0] * self.input1_w2) + 
                                   (input[1] * self.input2_w2) + 
                                   (input[2] * self.input3_w2) + 
                                   (input[3] * self.input4_w2))
        
        output_1 = ((inputs_to_top_hidden * self.output1_w1) + 
                    (inputs_to_bottom_hidden * self.output1_w2))
        
        output_2 = ((inputs_to_top_hidden * self.output2_w1) + 
                    (inputs_to_bottom_hidden * self.output2_w2))
        
        output_3 = ((inputs_to_top_hidden * self.output3_w1) + 
                    (inputs_to_bottom_hidden * self.output3_w2))
        
        output_4 = ((inputs_to_top_hidden * self.output4_w1) + 
                    (inputs_to_bottom_hidden * self.output4_w2))

        output_presoftmax = torch.stack([output_1, output_2, output_3, output_4])
        return(output_presoftmax)
    def configure_adam_optimizer(self):
        return Adam(self.parameters(), lr = 0.1)

    def training_step(self, batch, batch_idx):
        input_i, label_i = batch 
        output_i = self.forward(input_i)
        loss = self.loss_Function(output_i, label_i[0])
        return loss 

modelFromScratch = WordEmbeddingFromScratch()

print("Before optimization, the parameters are...")
for name, param in modelFromScratch.named_parameters():
    print(name, param.data)

data = {
    "w1": [modelFromScratch.input1_w1.item(),
           modelFromScratch.input2_w1.item(),
           modelFromScratch.input3_w1.item(),
           modelFromScratch.input4_w1.item()], 

    "w2": [modelFromScratch.input1_w2.item(),
           modelFromScratch.input2_w2.item(),
           modelFromScratch.input3_w2.item(),
           modelFromScratch.input4_w2.item()], 
    
    "token": ["My", "name", "is", "Esmail"], 
    "input": ["input1", "input2", "input3", "input4"]
} 

dataFrame = pd.DataFrame(data)
print(dataFrame)

sns.scatterplot(data = dataFrame, x= "w1", y = "w2")
plt.text(dataFrame.w1[0], dataFrame.w2[0], dataFrame.token[0], 
         horizontalalignment = 'left', 
         size = 'medium', 
         color = 'black',
         weight = 'semibold')
plt.text(dataFrame.w1[1], dataFrame.w2[1], dataFrame.token[1], 
         horizontalalignment = 'left', 
         size = 'medium', 
         color = 'black',
         weight = 'semibold')
plt.text(dataFrame.w1[2], dataFrame.w2[2], dataFrame.token[2], 
         horizontalalignment = 'left', 
         size = 'medium', 
         color = 'black',
         weight = 'semibold')
plt.text(dataFrame.w1[3], dataFrame.w2[3], dataFrame.token[3], 
         horizontalalignment = 'left', 
         size = 'medium', 
         color = 'black',
         weight = 'semibold')
plt.show()

# Training loop
# optimizer = Adam(modelFromScratch.parameters(), lr=0.1)
# num_epochs = 100
# for epoch in range(num_epochs):
#     total_loss = 0
#     for inputs, labels in dataloader:
#         optimizer.zero_grad()
#         outputs = modelFromScratch(inputs)
#         loss = modelFromScratch.loss_Function(outputs, labels[0])
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"Epoch {epoch+1}, Loss: {total_loss}")

# # After training
# print("After optimization, the parameters are...")
# for name, param in modelFromScratch.named_parameters():
#     print(name, param.data)

softmax = nn.Softmax(dim = 0)
print(torch.round(softmax(modelFromScratch(torch.tensor([[1., 0., 0., 0.]]))), decimals=2))
print(torch.round(softmax(modelFromScratch(torch.tensor([[0., 1., 0., 0.]]))), decimals=2))
print(torch.round(softmax(modelFromScratch(torch.tensor([[0., 0., 1., 0.]]))), decimals=2))
print(torch.round(softmax(modelFromScratch(torch.tensor([[0., 0., 0., 1.]]))), decimals=2))