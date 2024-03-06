# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 20:11:51 2024

@author: Cao Yu
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torcheval.metrics import R2Score

#0.To do list 10mins each
#1. Data division _done_
#2. Ramdom the data (Before data division) _done_
#3. Data Normalization 
#4. Data Denormalization
#5. Change the accuracy to R2 value formula _done
#6. Change the x value to x_train, y_train __done_modelmaybe noneed 
#7. Plotting: 1. R2 value _done_
#8. Plotting: 2. loss vs time _done_
#9. Plotting: 3. Predict (y_test) vs actual _done_
#10. Save the model (Save the directory)
#11. Save the plots 
#12. Load the model (fyi)
#13. Stop criteria


#1.Data loading

# load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('D:\study\PhD\Research Work\Model\DARNN\group1_tunnel31\Group_tunnel31.csv', delimiter=',') #改路径

#2.Preprocess
#2.1 Randomize the data


#2.2 Data division


#2.3 Data Preprocess
#3. Training
#3.1 model definition
class ann(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(8, 10)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(10, 10)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(10, 1)
        self.act_output = nn.Sigmoid()

    def forward(self, y):
        y = self.act1(self.hidden1(x))
        y = self.act2(self.hidden2(x))
        y = self.act_output(self.output(x))
        return y

model = ann()
print(model)

#3.2 Optimization definition

loss_fn = nn.MSELoss()  # meansquareerror
Optimizer = optim.Adam(model.parameters(), lr=0.001)


#3.3 training loop
batch_size = 10

#a. Training

def train(TrainData, model, loss_fn, Optimizer): 
    size = len(TrainData) #datset to bereplaced by training data size
        
    x_train=TrainData[:,0:8]
    y_train=TrainData[:,9]
    
    metric_train = R2Score()
    
    model.train()
        
    #train & compute loss
    y_pred1 = model(x_train)
    train_loss = loss_fn(y_pred1, y_train)
    ###Add one more row for R2 value, metric a 
    metric_train.update(y_pred1, y_train)
    metric_train.compute()
    print(f"Train R2 is {metric_train}")
    
    #backpropagation
    loss.backward() #backpropagation
    Optimizer.step() #update parameters
    Optimizer.zero_grad() #set the original gradient to be zero
    
    #add the calculation of r2
    return y_train, y_pred1, train_loss, metric_train
    
        
#b. Test (Performance part, no backpropagtion)

def test (TestData, model, loss_fn):
    size = len(TestData) #dataset to be replaced by test dataset or see if canbe removed
    num_batchs = len(TestData) ###
        
    x_test=TrainData[:,0:8]
    y_test=TrainData[:,9]
    
    metric_test = R2Score()
    
    model.eval()
    #test_loss, correct = 0, 0
    
    with torch.no_grad(): # no backpropagation
     ##to be raplaced by x_test, y_test data
     y_pred2 = model(x_test)
     test_loss = loss_fn(y_pred2, y_test) #test load叠加，c +=a, c=c+a
     metric_test.update(y_pred2, y_train)
     metric_test.compute()
  
    #test_loss /= num_bactches  #c/=a, is c=c/a
    print(f"Test Error: \n Test loss: {test_loss} \n Test R2 is {metric_test}")
    #f加了后面可以加公式
    return y_test, y_pred2, test_loss, metric_test
  

#c. Define the training loop

epochs = 100

# Add for plotting

#train_pred = []
#test_pred = []
trainingEpoch_loss = []
testEpoch_loss = []
train_R2 = []
test_R2 = []


for i in range(epochs):
    #ramdomized the data each time, and after that data division
    Data=np.random.shuffle(dataset)
    #convert to tensor
    #x = torch.tensor(x, dtype=torch.float32)
    #y = torch.tensor(y, dtype=torch.float32)

    #print(f"Shape of x: {x.shape}")
    #print(f"Shape of y: {y.shape}")
    
    #Data division to train data and test data by 7/3   
        
    Trainsize=torch.round(torch.tensor(len(Data)*0.7), decimals=0)-1
    Testsize=len(Data)-Trainsize
    
    a=Trainsize.to(torch.int32)
    b=a+1

    TrainData=Data[:a,:]
    TestData=Data[b:,:]


    print(f"Train sample is {TrainData}")
    print(f"Test sample is {TestData}")
    
    print(f"Epoch {t+1}\n-------") #\n表示换行
    
    #Preparation for plotting
    
    y_actual1, y_pred1, train_loss, R2_train = train(TrianData, model, loss_fn, Optimizer) #to be replaced by training dataset
    y_actual2, y_pred2, test_loss, R2_test = test(TestData, model, loss_fn) #to be replaced by test dataset
 
    #plotting
    #train_pred.append(y_pred1.item())
    #test_pred.append(y_pred2.item())
    trainingEpoch_loss.append(train_loss.item())
    testpoch_loss.append(test_loss.item())
    train_R2.append(R2_train.item())
    test_R2.append(R2_test.item()) 
    

print("Finish")

#5.Plotting
##Need to plotting the R2 value, error vs epoches, comparison of predicted value & actual value
#a. prdiction actual vs predict (2plts)
#a1. train
x_axis1 = range(1,Trainsize)
plt.plot(x_axis1, y_pred1, label='prediction_train')
plt.plot(x_axis1, y_actual1, label='actual')

#annotation
plt.title('Prediction(Train) vs Actual')
plt.xlabel('Sample')
plt.ylabel('Prediction')
#tick
plt.xticks(arange(0,Trainsize,10))
plt.legend(loc='best') ###
plt.show()

#a2. test (Model Performance)
x_axis1 = range(1,Testsizesize)
plt.plot(x_axis1, y_pred2, label='prediction_test')
plt.plot(x_axis1, y_actual2, label='actual')

#annotation
plt.title('Prediction(Test) vs Actual')
plt.xlabel('Sample')
plt.ylabel('Prediction')
#tick
plt.xticks(arange(0,Testsize,10))
plt.legend(loc='best') ###
plt.show()


#b.loss vs epochs
x_axis2 = range(1, epochs)
plt.plot(x_axis2, trainingEpoch_loss, label='train_loss')
plt.plot(x_axis2, testpoch_loss, label='test_loss')

#annotation
plt.title('Training and Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
#tick
plt.xticks(arange(0,epochs,10))
plt.legend(loc='best') ###
plt.show()

#c.R2 vs epochs
x_axis3 = range(1, epochs)
plt.plot(x_axis3, train_R2, label='train_R2')
plt.plot(x_axis3, test_R2, label='test_R2')

#annotation
plt.title('Training and Test R2')
plt.xlabel('Epochs')
plt.ylabel('R2')
#tick
plt.xticks(arange(0,epochs,10))
plt.legend(loc='best') ###
plt.show()

#6. Save
#a. Save the model (save the directory)
torch.save(model,r'D:\study\Research\J1\Data Experiment\Data Training\Pytorch\NN1')
torch.save(model.state_dict(),r'D:\study\Research\J1\Data Experiment\Data Training\Pytorch\NN1')

#b. Save the plots