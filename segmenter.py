import torch

# Detecting the free white space inside the image
def detector(images_dic:list , threshold , i):
  
  test_result = [torch.all(images_dic[i]['tensor'] , dim=0) , torch.all(images_dic[i]['tensor'] , dim=1)]
  dim_result = []
  for d,result in enumerate(test_result):
    detection_dic = {}

    solid = torch.where(result==False)[0]

    length = solid.diff() - 1

    where_in_length = torch.where(length >= threshold)[0]
    where = solid[where_in_length]+threshold
    detection_dic['dim'] = d
    detection_dic['solid'] = solid  if len(where) > 0 else None
    detection_dic['length'] = length if len(where) > 0 else None
    detection_dic['where'] = where if len(where) > 0 else torch.tensor([])

    if len(where) > 0: # even if one of the dimension has a value in where istrue will be True!
      images_dic[i]['istrue'] = True


    dim_result.append(detection_dic)
  images_dic[i]['data'] = dim_result

  return images_dic


def split_index_generator(images_dic:list,i:int) ->tuple:

  h,w = images_dic[i]['tensor'].shape

  dim_0_split_idx = (images_dic[i]['data'][0]['where']).tolist()
  dim_0_split_idx.insert(0,0)
  dim_0_split_idx.insert(len(dim_0_split_idx),w)
  dim_0_split_idx = torch.tensor(dim_0_split_idx).diff().tolist()

  dim_1_split_idx = (images_dic[i]['data'][1]['where']).tolist()
  dim_1_split_idx.insert(0,0)
  dim_1_split_idx.insert(len(dim_1_split_idx),h)
  dim_1_split_idx = torch.tensor(dim_1_split_idx).diff().tolist()

  return dim_0_split_idx , dim_1_split_idx


def tensor_splitter(images_dic:list , split_indices , image_tensor  , output:list):
  dim_0_idx, dim_1_idx = split_indices
  dim_0_splitted = torch.split(image_tensor , dim_1_idx , dim=0)
  for i in dim_0_splitted:
    images = torch.split(i , dim_0_idx , dim=1 )
    for i , g_images in enumerate(images):
        
        if g_images.shape == image_tensor.shape:
            images_dic.append({'tensor' : g_images , 'istrue' : False ,'data' : None })
            output.append({'tensor' : g_images , 'istrue' : False ,'data' : None })

        else:
            images_dic.append({'tensor' : g_images , 'istrue' : True ,'data' : None})

  return images_dic


def check_istrue(images_dic):
  x = [image['istrue'] for image in images_dic]
  if True in x:
    return True

def segmenter(main_image_tensor:torch.Tensor):

    # initialize the dic
    images_dic = [{'tensor' : None , 'istrue' : False ,'data' : None}]
    
    # we can use GrayModifier() transform here if it's required!

    # add the main image
    images_dic[0] = {'tensor' : main_image_tensor, 'istrue' : True ,  'data' : None}

    threshold = 2
    i = 0
    output = []

    while check_istrue(images_dic): 
        if images_dic[i]['istrue'] == True:
            images_dic = detector(images_dic , threshold , i)
            split_indices = split_index_generator(images_dic,i)

            check_istrue(images_dic)
            images_dic[i]['istrue'] = False
            images_dic = tensor_splitter(images_dic , split_indices , images_dic[i]['tensor'] , output)

        i+=1
    return output

