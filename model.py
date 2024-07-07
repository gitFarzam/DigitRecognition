import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
from data import train_set,test_set


class DigitDetectionNet(nn.Module):
    def __init__(self, in_channels , out_channels , k_size , k_stride , p_ksize , p_stride , n_out):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels= in_channels, out_channels=out_channels , kernel_size=k_size , stride = k_stride),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=p_ksize , stride=p_stride)
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=2*13*13 , out_features=120),
            nn.Linear(in_features=120 , out_features=n_out)
        )

    def forward(self, x:torch.Tensor)-> torch.Tensor:
        y = self.conv1(x) #; print('conv1 shape',y.shape)
        y = y.view(-1,2*13*13) #; print('reshape : ',y.shape)
        y = self.fc(y) #; print('output shape: ',y.shape)
        return y

def accuracy(label:torch.Tensor ,output:torch.Tensor) -> int:
    #print('output: ',output)
    output = torch.softmax(output , dim=1) ; #print('output after softmax: ',output)
    output = torch.argmax(output , dim=1); #print('output after argmax: ',output)

    return torch.eq(input=label , other=output).sum().item()

def training( epochs:int , train_loader : torch.utils.data.DataLoader, test_loader : torch.utils.data.DataLoader , model : nn.Module , lr:float)-> tuple:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters() , lr=lr)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        accurates = 0
        for batch , (inputs,labels) in enumerate(train_loader):
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss
            accurates = accurates + accuracy(labels,outputs)
            # print('labels: ', labels)
            # print('outpus: ' , outputs)
            # print('accurates : ' , accurates)
        acc_train = accurates/len(train_set)
        loss_per_batch = total_loss/len(train_loader)

        model.eval()
        accurates_test = 0
        with torch.inference_mode():
            for batch , (inputs,labels) in enumerate(test_loader):
                outputs = model(inputs)
                accurates_test = accurates_test + accuracy(labels,outputs)
            acc_test = accurates_test/len(test_set)

        print(f" loss (per batch): {loss_per_batch:.5f} | acc_train : {acc_train*100:.2f} | acc_test : {acc_test*100:.2f} ")

    return loss_per_batch , acc_train , acc_test


