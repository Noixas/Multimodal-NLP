import json
from types import SimpleNamespace
import pickle

'''
    Initially we save the gender and race probabilities of images in an array such that the first row of the array was 
    corresponding to the first row in the .jsonl file. However, this caused problems during resampling where we shuffle
    and upsample images. Therefore, this scripts converts the array to a dictionary that maps the img_id to
    its probabilities. We decided to write an additional script instead of modifying and rerunning `gender_race_predictions.py`
    as that files take several hours to run. Conversely, the script `add_img_id_to_gender_race_probs.py` takes less than a second.
'''

def add_img_id_to_gender_race_probs(args):
    filepath = args.filepath
    name = filepath.split("/")[-1].split(".")[0]

    with open(filepath) as f:
            json_list = [json.loads(json_dict) for json_dict in f.readlines()]

    data = SimpleNamespace(ids=None, gender_race_probs_dict={})
    data.ids = [example["id"] for example in json_list]

    with open(f'dataset/gender_race_probs/{name}_gender_race_probs.pickle', 'rb') as f:
            gender_race_probs = pickle.load(f)
    
    for i, img_id in enumerate(data.ids):
        data.gender_race_probs_dict[img_id] = gender_race_probs[i]
    
    with open(f'dataset/gender_race_probs/{name}_gender_race_probs_dict.pickle', 'wb') as f:
        pickle.dump(data.gender_race_probs_dict, f)

    # Test if the file was modified correctly
    
    with open(f'dataset/gender_race_probs/{name}_gender_race_probs_dict.pickle', 'rb') as f:
        saved_dict = pickle.load(f)
    
    print("### Check saved dict")
    print("saved dict type ", type(saved_dict))
    for key in saved_dict:
        print("saved dict first img: img_id", key, "img probs", saved_dict[key])
        break



if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str,
                        help="Path to jsonl file of dataset", required=True)

    args = parser.parse_args()

    add_img_id_to_gender_race_probs(args)                        
    