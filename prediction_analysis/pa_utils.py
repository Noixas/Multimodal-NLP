import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import os, json


pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)


def object_id_to_name(object_vocab, object_id):
    return object_vocab[object_id]


def display_img(img_id):
    img_path = "../dataset/img/"
    img = mpimg.imread(f"{img_path}/{id_to_img(img_id)}")
    imgplot = plt.imshow(img)
    plt.show()
    
def id_to_img(img_id):
    return str(img_id).zfill(5) + ".png"

def read_data(split_file):
    with open(os.path.join("..","dataset",split_file+".jsonl")) as f:
        json_list = [json.loads(json_dict) for json_dict in f.readlines()]
    return pd.DataFrame(json_list)

def read_objects(split_df):
    try:
        from collections import defaultdict
        objects = defaultdict(list)
        for img_id in split_df.id:
            img_feat_info = np.load(os.path.join("..","dataset","own_features", str(img_id).zfill(5) + "_info.npy"), 
                                    allow_pickle=True).item()

            objects["object_id"] += list(img_feat_info["objects_id"])
            objects["object_conf"] += list(img_feat_info["objects_conf"])
            objects["image_id"] += [img_id] * len(img_feat_info["objects_id"])
            objects["x1"] += list(img_feat_info["bbox"][:,0])
            objects["y1"] += list(img_feat_info["bbox"][:,1])
            objects["x2"] += list(img_feat_info["bbox"][:,2])
            objects["y2"] += list(img_feat_info["bbox"][:,3])
        
        return pd.DataFrame(objects).reset_index(drop=True)       
    except NameError as e:
        print(e)
        print("* Please read the dataset with read_data() before calling read_objects() *")


def compare_models_to_base(base_name, extensions):
    # read dev set confounders
    confounders = pd.read_csv("dev_data_conf_labels_fixed.csv")
    
    # read dev set gender and race probs
    gender_race_probs = pd.read_pickle("../dataset/gender_race_probs/dev_seen_gender_race_probs_dict.pickle")
    
    # gender and race labels
    gender_labels = ["female", "male"]
    race_labels = ["asian", "indian", "black", "white", "middle eastern", "latino hispanic"]
    
    # read baseline preds
    dev_preds = pd.read_csv(f"../model_checkpoints/{base_name}/meme_dev_seen_preds.csv")
    dev_preds = pd.merge(dev_preds, confounders[["id", "confounder_type"]], on="id")
    
    # for each extension, print images that became correctly classified and images that became incorrectly classified
    for run_setup, run_name in extensions.items():
        # add labels of a new model
        dev_preds["label "+run_setup] = pd.read_csv(f"../model_checkpoints/{run_name}/meme_dev_seen_preds.csv")["label"]
        
        # get memes that became correct
        print("### Images that became classified correctly with "+run_setup+" ###")
        became_correct = dev_preds[(dev_preds["label"]!=dev_preds["gt"]) & (dev_preds["label "+run_setup]==dev_preds["gt"])]
        for i, img_id in enumerate(list(became_correct.id.values)):
            # if the model contains gender and probabilities 
            if run_setup == "base + gender and race probs":
                probs = gender_race_probs[img_id]
                
                # if face detected, print dominant gender and race
                if probs[0] == 0 and probs[1] == 0:
                    print("No face detected")
                else:
                    dominant_gender = gender_labels[np.argmax(probs[:2])]
                    dominant_race = race_labels[np.argmax(probs[2:])]
                    print(dominant_gender, dominant_race)
                    
            print("Ground truth:", became_correct.iat[i,3])
            display_img(img_id)
        
        # get memes that became incorrect
        print("### Images that become classified incorrectly with "+run_setup+" ###")
        became_incorrect = dev_preds[(dev_preds["label"]==dev_preds["gt"]) & (dev_preds["label "+run_setup]!=dev_preds["gt"])]

        for i, img_id in enumerate(list(became_incorrect.id.values)):
            if run_setup == "base + gender and race probs":
                probs = gender_race_probs[img_id]
                
                if probs[0] == 0 and probs[1] == 0:
                    print("No face detected")
                else:
                    print(gender_labels[np.argmax(probs[:2])], race_labels[np.argmax(probs[2:])])
                    
            print("Ground truth:", became_incorrect.iat[i,3])
            display_img(img_id)
    
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import metrics
import seaborn as sns
# conf_type describes which modality stays the same i.e. "text" means that the text is the same as the hateful meme
# but the image was changed - which means its an image confounder

def plot_confusion_matrices(runs):
    # read dev set confounders
    confounders = pd.read_csv("dev_data_conf_labels_fixed.csv")
    
    for run_setup, run_name in runs.items():
        dev_preds = pd.read_csv(f"../model_checkpoints/{run_name}/meme_dev_seen_preds.csv")
        dev_preds = pd.merge(dev_preds, confounders[["id", "text", "confounder_type"]], on="id")

        # confusion matrix
        cm = confusion_matrix(dev_preds["gt"], dev_preds["label"])
        ax = plt.subplot()
        sns.set(font_scale=3.0) # Adjust to fit
        sns.heatmap(cm, annot=True, ax=ax, cmap="Blues", fmt="g", cbar=False, square=True)  

        # Labels, title and ticks
        label_font = {'size':'18'}  # Adjust to fit
        ax.set_xlabel('Predicted labels', fontdict=label_font)
        ax.set_ylabel('Observed labels', fontdict=label_font)

        title_font = {'size':'21'}  # Adjust to fit
        ax.set_title(run_setup, fontdict=title_font)

        ax.tick_params(axis='both', which='major', labelsize=15)  # Adjust to fit
        ax.xaxis.set_ticklabels(['0', '1']);
        ax.yaxis.set_ticklabels(['0', '1']);
        plt.show()
        
def count_subclasses(runs, print_results=False):
    # read dev set confounders
    confounders = pd.read_csv("dev_data_conf_labels_fixed.csv")
    
    subclasses = []
    for run_setup, run_name in runs.items():
        dev_preds = pd.read_csv(f"../model_checkpoints/{run_name}/meme_dev_seen_preds.csv")
        dev_preds = pd.merge(dev_preds, confounders[["id", "text", "confounder_type"]], on="id")
        
        # count how many confounders are correctly classified
        img_confs = dev_preds[(dev_preds["gt"]==0) & (dev_preds["confounder_type"]=="text")]
        txt_confs = dev_preds[(dev_preds["gt"]==0) & (dev_preds["confounder_type"]=="image")]
        other = dev_preds[(dev_preds["gt"]==0) & (dev_preds["confounder_type"]=="Other")]

        correct_img_conf = img_confs[img_confs["label"] == 0]
        correct_txt_conf = txt_confs[txt_confs["label"] == 0]
        correct_other = other[other["label"]==0]

        hateful_img_confs = dev_preds[(dev_preds["gt"]==1) & (dev_preds["confounder_type"]=="text")] 
        hateful_txt_confs = dev_preds[(dev_preds["gt"]==1) & (dev_preds["confounder_type"]=="image")]
        hateful_other = dev_preds[(dev_preds["gt"]==1) & (dev_preds["confounder_type"]=="Other")]
        hateful_both = dev_preds[(dev_preds["gt"]==1) & (dev_preds["confounder_type"]=="both")]

        correct_hateful_img_conf = hateful_img_confs[hateful_img_confs["label"] == 1]
        correct_hateful_txt_conf = hateful_txt_confs[hateful_txt_confs["label"] == 1]
        correct_hateful_other = hateful_other[hateful_other["label"] == 1]
        correct_hateful_both = hateful_both[hateful_both["label"] == 1]
        
        if print_results:
            print(f"### {run_setup} ###")
            print(f"Correctly classified {correct_img_conf.shape[0]} out of {img_confs.shape[0]} image confounders")
            print(f"Correctly classified {correct_txt_conf.shape[0]} out of {txt_confs.shape[0]} text confounders")  
            print(f"Correctly classified {correct_other.shape[0]} out of {other.shape[0]} other non-hateful")

            print(f"Correctly classified {correct_hateful_img_conf.shape[0]} out of {hateful_img_confs.shape[0]} hateful memes with corresponding image conf")
            print(f"Correctly classified {correct_hateful_txt_conf.shape[0]} out of {hateful_txt_confs.shape[0]} hateful memes with corresponding text confounders")
            print(f"Correctly classified {correct_hateful_other.shape[0]} out of {hateful_other.shape[0]} hateful memes without corresponding confounders")
            print(f"Correctly classified {correct_hateful_both.shape[0]} out of {hateful_both.shape[0]} hateful memes with both corresponding confounders")

        subclasses.append({"image confounders":correct_img_conf.shape[0] / img_confs.shape[0],
                        "text confounders":correct_txt_conf.shape[0] / txt_confs.shape[0],
                        "other non-hateful":correct_other.shape[0] / other.shape[0],
                        "hateful w/ image confounders": correct_hateful_img_conf.shape[0] / hateful_img_confs.shape[0],
                        "hateful w/ text confounders": correct_hateful_txt_conf.shape[0] / hateful_txt_confs.shape[0],
                        "hateful w/o confounders": correct_hateful_other.shape[0] / hateful_other.shape[0],
                        "hateful w/ both confounders": correct_hateful_both.shape[0] / hateful_both.shape[0]
                       })
    subclasses_df = pd.DataFrame(subclasses, index = [run_setup.replace("base + ","") for run_setup in runs])
    return subclasses_df
    

def plot_aurocs(runs):
    displays = []
    fig, ax = plt.subplots()
    
    for run_setup, run_name in runs.items():
        dev_preds = pd.read_csv(f"../model_checkpoints/{run_name}/meme_dev_seen_preds.csv")
        fpr, tpr, thresholds = metrics.roc_curve(dev_preds['gt'], dev_preds['proba'])
        roc_auc = metrics.auc(fpr, tpr)
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=run_setup)
        display.plot(ax)  
        
    plt.show()

def plot_subclass_accs(subclasses_df):
    subclasses_df.T.plot(kind="bar",figsize=(6, 4))
    plt.title("Accuracies for different subclasses of non-hateful memes", fontsize=12)
    plt.legend(loc="lower right", framealpha=1, fontsize=10) # bbox_to_anchor=(1.0, 1.40)
    plt.xticks(rotation=0, fontsize=10)
    plt.savefig("subclass_accs.png", dpi=500)
    plt.show()

