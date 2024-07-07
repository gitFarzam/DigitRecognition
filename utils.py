import os
import csv

def img_label_tolist(path:str , subset:int) -> list:
    dir = os.listdir(path)
    junk = ".DS_Store"
    # remove unwanted files
    os.remove(f"{path}/{junk}") if ".DS_Store" in dir else print('no junk in main dir')
    result = [[i,os.listdir(path=f"{path}/{i}/{i}")[:subset]] for i in os.listdir(path=f"{path}")]
    result.remove(junk) if junk in result else print("no junk in list")
    return result

def csv_writer(img_label:list) -> None:
    with open('data/annotations_file.csv', 'w') as csvfile:
        writer = csv.writer(csvfile , delimiter=',')
        writer.writerow(['image','label'])
        for data in img_label:
            for img in data[1]:
                writer.writerow([f"{data[0]}/{data[0]}/{img}" , data[0]])
                # print([f"{data[0]}/{img}" , data[0]])


