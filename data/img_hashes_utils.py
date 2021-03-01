import json
import imagehash
from PIL import Image
from tqdm import tqdm
tqdm.pandas()
import json
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import itertools
from tqdm.auto import trange, tqdm

SEED = 2021
DATA_MULT = 3 # Used to define how many times more we the data to increase
main_path = "dataset/"
train_path = "train.jsonl"
dev_path = "dev_seen.jsonl"
test_path = "test_seen.jsonl"
pred_path = "model_checkpoints/Best submission so far/"
dev_pred_filename = "meme_dev_seen_preds.csv"

def get_json_list(filepath):
    json_list = pd.read_json(main_path+filepath, lines=True) 
    return json_list
def display_img(img_id):
    img_path = "dataset/img/"
    img = mpimg.imread(f"{img_path}/{id_to_img(img_id)}")
    imgplot = plt.imshow(img)
    plt.show()
def display_pair_img(img_id_1,img_id_2):
    plt.figure()
    #subplot(r,c) provide the no. of rows and columns
    f, axs = plt.subplots(1,2) 

    img_path = "dataset/img/"
    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    axs[0].imshow(mpimg.imread(f"{img_path}/{id_to_img(img_id_1)}"))
    axs[1].imshow(mpimg.imread(f"{img_path}/{id_to_img(img_id_2)}"))   
    axs[0].set_title(str(img_id_1))
    axs[1].set_title(str(img_id_2))
    plt.show()
def id_to_img(img_id):
    return str(img_id).zfill(5) + ".png"

    

def get_hashes_from_df_id(df_data):
    return[(i,imagehash.dhash(Image.open("dataset/img/"+id_to_img(i) ))) for i in tqdm(df_data.id)] # Dev data hash

#Read hashes from file
def read_hashed_from_file(filename='hash_imgs_dhash_default_filename.json',filepath='dataset/upsampling/'):
    with open(filename,'r') as filehandle:
        # print(filehandle.readline)
        similar_ids = json.load(filehandle)
    #Convert back to hashes
    hash_imgs_train = [(pair[0],imagehash.hex_to_hash(pair[1])) for pair in similar_ids]
    return hash_imgs_train

def save_hashes_to_file(hashes, filename='hash_imgs_dhash_default_filename.json',filepath='dataset/upsampling/'):
    saving_hash = [(hash_tuple[0],str(hash_tuple[1])) for hash_tuple in hashes] # Dev data hash

    with open(filename, 'w') as filehandle:
        json.dump(saving_hash, filehandle)
def get_pairs_repeated_images(hash_array, threshold=8):
    similar_ids = []
    for a, b in tqdm(itertools.combinations(hash_array, 2)):
        crop_diff = a[1] - b[1]

        if crop_diff < threshold:
            # print("\nHash has {} percentage difference ".format(crop_diff))
            # print("ID 1:",str(a[0]),"Labeled as:",int(train_data[train_data.id==a[0]]['label']))
            # print("ID 2:",str(b[0]),"Labeled as:",int(train_data[train_data.id==b[0]]['label']))
            # print("ID of match:",str(id_img))
            # display_pair_img(a[0],b[0])
            similar_ids.append((a[0],b[0]))
    
    print("\nDONE ",str(len(similar_ids))," repeated images found\n")
    return similar_ids

def save_pairs_similar_imgs(similar_ids, filepath):
    with open(filepath, 'w') as filehandle:
        json.dump(similar_ids, filehandle)
def load_pairs_similar_imgs(filepath):
    with open(filepath,'r') as filehandle:
        similar_ids = json.load(filehandle)
    return similar_ids

#Preview similar pairs
def preview_x_amount_pairs(list_ids_pairs, amount_to_preview = 2):    
    for sim_id in list_ids_pairs[:amount_to_preview]:
        display_pair_img(sim_id[0],sim_id[1])
def get_ids_repeated_imgs(list_ids_pairs):
    ids_set = set()
    for tupl in list_ids_pairs:
        ids_set.add(tupl[0])
        ids_set.add(tupl[1])
    return list(ids_set)


def save_confounders_ids(confounder_ids,filepath):
    with open(filepath, 'w') as filehandle:
        json.dump(confounder_ids, filehandle)
def load_confounders_ids(filepath):
    with open(filepath,'r') as filehandle:
        confounder_ids = json.load(filehandle)
    return confounder_ids