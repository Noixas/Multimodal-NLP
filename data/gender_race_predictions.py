import json
import os
import numpy as np
import sys

'''
import deepface from a cloned git repo assuming your directory looks like that:

> deepface
> Multimodal-NLP
  |
  > data
    - gender_race_predictions.py

'''
sys.path.append(os.path.join(sys.path[0], '../../deepface/'))
from deepface import DeepFace
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
def get_gender_race_predictions(args):
    with open(args.filepath) as f:
        json_list = [json.loads(json_dict) for json_dict in f.readlines()]

    img_paths = [example["img"] for example in json_list]
    img_ids = [example["id"] for example in json_list]

    assert len(img_paths) == len(img_ids) 
    img_preds = np.zeros((len(json_list), 8))
    # imgs_with_people_idxs = []
    # imgs_with_people_paths = []
    # for i, example in enumerate(json_list):
    #     img_path = example["img"]
    #     img_id = str(example["id"]).zfill(5)
    #     info_np_file = os.path.join(args.feature_dir, img_id + "_info.npy")
    #     assert os.path.exists(info_np_file), f"File '{info_np_file}' does not exist"

    #     img_feat_info = np.load(os.path.join(args.feature_dir, img_id + "_info.npy"), allow_pickle=True).item()
    #     if does_img_contain_people(img_feat_info["objects_id"]):
    #         imgs_with_people_idxs.append(i)
    #         imgs_with_people_paths.append(img_path)
    
    # prepare a list of image paths for bulk prediction
    img_paths = ["../dataset/" + img_path for img_path in img_paths]       
    # make predictions for all images with people - preds is a list where each element is a matrix of shape (num_people_in_img, 8)
    print("Detect gender and ethnicity", img_paths[0])
    preds = DeepFace.analyze(img_paths, actions = ['gender', 'race'], enforce_detection = False)
    # retrieve probabilities and convert to array
    preds = [img_probs["gender"] + img_probs["race"] for img_probs in preds]
    preds = np.array(preds)
    
    assert preds.shape==(len(img_paths), 8)

    return preds

    

def does_img_contain_people(object_ids):
    person_ids = []
    for person_id in person_ids:
        if person_id in object_ids:
            return True
    return False


if __name__ == '__main__':
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str,help="Path to jsonl file of dataset", required=True)
    parser.add_argument('--feature_dir', type=str, help='Directory containing image features', required=True)
    args = parser.parse_args()

    split = args.filepath.split("\\")[-1].split(".")[0]
    print("split", split)

    img_preds = get_gender_race_predictions(args)

    
    with open(f'{split}_gender_race_probs.pickle', 'wb') as f:
        pickle.dump(img_preds, f)

    
