import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import os, json
from pa_utils import *

class ObjectsLoader():
    def __init__(self, objects_file):
        super().__init__()
        self._load_objects_df(objects_file)
        self._load_object_vocab()
    
    def _load_object_vocab(self):
        with open("../objects_vocab.txt") as f:
            vocab_file = f.read().split("\n")   

        self.object_vocab = {i:name for i, name in enumerate(vocab_file)}

    def _load_objects_df(self, objects_file):
        if os.path.exists(objects_file):
            objects_df = pd.read_csv(objects_file)
        else:
            if objects_file == "train_dev_objects.csv":
                train, dev = [read_data(split_file) for split_file in ["train", "dev_seen"]]

                train_objects = read_objects(train)
                dev_objects = read_objects(dev)
            
                objects_df = pd.concat([train_objects, dev_objects])
            else:
                assert False, "Wrong file"

            objects_df.to_csv(objects_file, index=False)
        
        self.objects_df = objects_df

    def show_img_bboxes(self, img_id, bbox_coords, include_caption = True):
        img_path = "../dataset/img/"
        img = mpimg.imread(f"{img_path}/{id_to_img(img_id)}")

        # Create figure and axes
        fig, ax = plt.subplots()
        ax.imshow(img)
        
        img_df = self.objects_df[self.objects_df["image_id"]==img_id]
        
        for i, (x1, y1, x2, y2) in enumerate(bbox_coords):
            bbox = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color="black")
            ax.add_patch(bbox)
        plt.show()
    
    def get_top3_coords(self, img_id):
        top3 = self.objects_df[self.objects_df["image_id"]==img_id].groupby("object_id").apply(get_top3)
        bbox_coords = top3[["x1", "y1", "x2", "y2"]].values
        return bbox_coords

    def get_all_coords(self, img_id):
        top3 = self.objects_df[self.objects_df["image_id"]==img_id]
        bbox_coords = top3[["x1", "y1", "x2", "y2"]].values
        return bbox_coords

    def get_most_conf_coords(self, img_id, conf_thresh=0.5):
        top3 = self.objects_df[self.objects_df["image_id"]==img_id][self.objects_df["object_conf"]>conf_thresh]
        bbox_coords = top3[["x1", "y1", "x2", "y2"]].values
        return bbox_coords

        
    def get_img_object_names(self, img_id):
        img_objects = self.objects_df[self.objects_df["image_id"]==img_id]
        img_obj_ids = list(img_objects["object_id"].values)
        return [object_id_to_name(self.object_vocab, object_id) for object_id in img_obj_ids]

def get_top3(x):
    return x.sort_values("object_conf", ascending=False).head(3)

