import torch
from model import  DigitDetectionNet , training
from data import input_cc , num_labels,inputs_sample , train_loader , test_loader
from configuration import model_configuration,data_configuration
from torchsummary import summary

print('subset: ', data_configuration['images_subset'])

model = DigitDetectionNet(in_channels=input_cc , out_channels=2 , k_size=2,k_stride=1,p_ksize=2,p_stride=2,n_out=num_labels )

FILE = f"models_archive/model_{data_configuration['images_subset']}.pth"
model.load_state_dict(torch.load(FILE , map_location=torch.device('cpu')))

#Model Summary
summary(model=model , input_size=(1, 28, 28))

"""test an input sample in the model"""
# model(inputs_sample)

loss_per_batch , acc_train , acc_test = training(epochs=model_configuration['epochs'], train_loader=train_loader , test_loader=test_loader , model=model , lr=model_configuration['lr'])


torch.save(model.state_dict() , FILE)

"""
1. use classes in functions input and output √
2. change names of im to image and others √
3. delete redundant statements
4. remove redundant imports
5. use model accuracy plots
6. fix indents
7. showing images beside each other √
8. change samples, provide new and.. use internet to find new samples
9. use Graymodifier inside the segmenter file √
"""