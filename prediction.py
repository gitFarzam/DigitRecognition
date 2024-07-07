import torch
from torchvision import transforms
from model import  DigitDetectionNet
from data import input_cc,num_labels
from PIL import Image
import matplotlib.pyplot as plt
from segmenter import segmenter
from transform import SquareCrop, EdgeCrop, GrayModifier
from anomaly import white_detector,black_detector

# Build the model
model = DigitDetectionNet(in_channels=input_cc , out_channels=2 , k_size=2,k_stride=1,p_ksize=2,p_stride=2,n_out=num_labels )

# Load the trained model
FILE = f"models_archive/model_-1.pth"
model.load_state_dict(torch.load(FILE , map_location=torch.device('cpu')))
model.eval()

# predict a sample from testing set
def test_set_sample():
    from data import test_set
    i = 1800
    sample_input  , sample_label = test_set[i]
    print(f"indexed Serie from df: {test_set.img_labels.iloc[i]} , \nsample input: {sample_input.shape},sample labell: {sample_label.shape }, sample label: {sample_label}" )

    with torch.inference_mode():
        prediction = model(sample_input)
        scores = torch.softmax(prediction,dim=1)
        argmaxoutput = torch.argmax(scores , dim=1)
        print(f"Actual label : {sample_label} | \nscores : {scores} | \nwinner: {argmaxoutput}")

# test_set_sample()


# predict a new sample
def new_sample(image_path:str):

    main_image = Image.open(image_path).convert('RGB')
    main_image.show()



    t_1 = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Grayscale()
        ])

    t_2 = transforms.Compose([
        EdgeCrop(),
        SquareCrop(),
        transforms.Resize(size=28)
        ])

    image = t_1(main_image)

    gm = GrayModifier(0.3)
    image = gm(image)
    plt.imshow(image.squeeze(0) , cmap='gray');plt.show()

    segmented_images = segmenter(image.squeeze(dim=0))

    fig, ax = plt.subplots(ncols=6,nrows=int(len(segmented_images)/6)+1,figsize=(14,30))
    ax = ax.flatten()
    for i,image in enumerate(segmented_images):

        with torch.inference_mode():

            image = image['tensor']
            if white_detector(image):
                
                image = t_2(image.unsqueeze(dim=0))
                if black_detector(image):

                    prediction = model(image)

                    scores = torch.softmax(prediction.squeeze(0),dim=0)
                    scores_top_3 = sorted(list(enumerate([round(v.item(),4) for v in scores])) , key=lambda x:x[1] , reverse=True)[:3]

                    print('Top 3 Scores belongs to: ',scores_top_3)
                    
                    argmaxoutput = torch.argmax(scores , dim=0)
                    # print(f"\nscores : {scores}")
                    print(f"Winner: {argmaxoutput} \n----------------------")

                    title = f"{argmaxoutput.item()} - S: {int(scores_top_3[0][1]*100)}%"
                else:
                    print("black anomaly")
                    title = "black anomaly"
            else:
                print('white anomaly')
                title = "white anomaly"

            ax[i].imshow(image.squeeze(0), cmap='gray')
            ax[i].set_title(label=f"P: {title}",fontdict={'color':'red' , 'backgroundcolor' : 'white'} )
    # fig.suptitle(f"Prediction digits for:\n{image_path}" , fontsize=20)
    else:
        [ax.set_visible(False) for ax in ax.flatten()[i+1:]]
    plt.tight_layout()
    
    plt.show()


image_path = 'data/sample/numbers6.jpg'
new_sample(image_path)
