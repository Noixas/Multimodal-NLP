from tokenize import Double
import torch
import torch.utils.data as data
import numpy as np
import os
from PIL import Image
from types import SimpleNamespace
import logging
import matplotlib.pyplot as plt
import wandb
import pandas as pd
try:
    from utils.utils import get_attention_mask, get_gather_index
except ModuleNotFoundError as e:
    import sys
    sys.path.append(os.path.join(sys.path[0], '..'))
    from utils.utils import get_attention_mask, get_gather_index

import json
from torch.nn.utils.rnn import pad_sequence

logging.basicConfig(format='%(asctime)s : %(levelname)s - %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO)
logger = logging.getLogger('MemeDatasetLog')


class MemeDataset(data.Dataset):

    def __init__(self,
                 filepath: str,
                 feature_dir: str = None,
                 text_preprocess=None,
                 text_padding=None,
                 compact_batch: bool = True,
                 confidence_threshold: float = 0.0,
                 filter_text=False,
                 upsample_multiplier=0):
        """
        Inputs:
            filepath - Filepath to the ".jsonl" file which stores a list of all data points.
            feature_dir - Directory containing image features.
            text_preprocess - Function to execute after loading the text. Takes as input a list of all meme text in the dataset.
                              Is expected to return a list of processed text elements.
            text_padding - Function to apply for padding the input text to match a batch tensor.
            compact_batch - If True, batches with text and image will be compacted without padding between them
            confidence_threshold - Threshold of object confidence under which bounding boxes are ignored
        """
        super().__init__()
        self.filepath = filepath
        self.name = filepath.split("/")[-1].split(".")[0]
        self.feature_dir = feature_dir
        self.text_preprocess = text_preprocess
        self.text_padding = text_padding
        self.compact_batch = compact_batch
        self.confidence_threshold = confidence_threshold
        self.return_ids = True
        self.filter_text = filter_text
        self.upsample_multiplier = upsample_multiplier
        print("filter text", self.filter_text)

        self._prepare_data_list()

    def upsample_confounders(self):
        SEED = 2021
        multiplier = self.upsample_multiplier
        # read data
        train_data_df = pd.read_json(self.filepath, lines=True)
        # Get rows with duplicated text.
        duplicated_text = train_data_df[train_data_df['text'].duplicated(
            keep=False)]
        new_suffix = ''

        #################################
        #Upsample not hateful only - option 1
        #################################
        # # Explicitely get the text of the hateful memes, since some duplicated text appears only in non hateful memes.
        # text_confounders = duplicated_text.loc[(
        #     duplicated_text['label'] == 1)]['text']
        # # Get rows of non hateful memes that contain duplicated text.
        # rows_label_0 = duplicated_text.loc[(duplicated_text['label'] == 0)]
        # # Get rows with label 0 that their text appeared in hateful memes.
        # rows_confounders = rows_label_0.loc[rows_label_0['text'].apply(
        #     (lambda x: any(item for item in text_confounders if item in x)))]

        # len_rows_confounders = len(rows_confounders)
        
        # # Create an upsample of data by sampling with replacement and reseting index
        # rows_confounders_upsampled = rows_confounders.sample(
        #     n=len_rows_confounders*multiplier, replace=True, random_state=SEED).reset_index(drop=True)
        # print("Confounders upsampled by", str(multiplier), "times. \n From", str(
        #     len_rows_confounders), " samples to", str(len(rows_confounders_upsampled)))
        # print("Upsample not hateful only - option 1")
        



        #################################  #BEST RESULTS
        #Upsample both parts of confounders, hateful and non hateful - option 2
        #################################
        # Explicitely get the text of the hateful memes, since some duplicated text appears only in non hateful memes.
        text_confounders_label1 = duplicated_text.loc[(
            duplicated_text['label'] == 1)]
        text_confounders = text_confounders_label1['text']
         # Get rows of non hateful memes that contain duplicated text.
        rows_label_0 = duplicated_text.loc[(duplicated_text['label'] == 0)]
        # Get rows with label 0 that their text appeared in hateful memes.
        rows_confounders = rows_label_0.loc[rows_label_0['text'].apply(
            (lambda x: any(item for item in text_confounders if item in x)))]

        len_rows_confounders = len(rows_confounders)
        len_rows_confounders_l1 = len(text_confounders_label1)
        
        # Create an upsample of data by sampling with replacement and reseting index
        rows_confounders_upsampled_label_0 = rows_confounders.sample(
            n=len_rows_confounders*multiplier, replace=True, random_state=SEED).reset_index(drop=True)
        rows_confounders_upsampled_label_1=text_confounders_label1.sample(
            n=len_rows_confounders_l1*multiplier, replace=True, random_state=SEED).reset_index(drop=True)

        rows_confounders_upsampled = pd.concat([rows_confounders_upsampled_label_0,rows_confounders_upsampled_label_1])

        print("Confounders upsampled by", str(multiplier), "times. \n From", str(
            len_rows_confounders+len_rows_confounders_l1), " samples to", str(len(rows_confounders_upsampled)))
        print("Upsample both parts of confounders, hateful and non hateful - option 2")


        
        
        #################################
        #Upsample both hateful and not hateful duplicated text memes - option 3
        #################################
        # len_rows_confounders = len(duplicated_text)        
        # # Create an upsample of data by sampling with replacement and reseting index
        # rows_confounders_upsampled = duplicated_text.sample(
        #     n=len_rows_confounders*multiplier, replace=True, random_state=SEED).reset_index(drop=True)
        # print("Confounders upsampled by", str(multiplier), "times. \n From", str(
        #     len_rows_confounders), " samples to", str(len(rows_confounders_upsampled)))
        # print("Upsample both hateful and not hateful duplicated text memes - option 3")





        #################################
        #Augment data by changing text of toxic memes for random text of non toxic memes - Option A, use together with options 1, 2 or 3
        #################################
        # DATA_MULT_AUGM = 2
        # #Get label 1 data
        # label_1_rows = train_data_df.loc[train_data_df.label==1]
        # sample_amount = len(label_1_rows)*DATA_MULT_AUGM
        # #Upsample toxic memes
        # resample_label_1 = label_1_rows.sample(n=sample_amount,replace=True,random_state=SEED).reset_index(drop=True)
        
        # #Get non toxic memes
        # label_0_rows = train_data_df.loc[train_data_df.label==0]
        # #Get the text of non toxic memes
        # label_0_unique_txt = label_0_rows['text'].unique()
        # #Upsample non toxic texts to match the amount of toxic memes that will be detoxified
        # resample_label_0_txt = pd.DataFrame(label_0_unique_txt,columns=['text']).sample(n=sample_amount,replace=True,random_state=SEED).reset_index(drop=True)
        # #Change the text of toxic memes for the one of non toxic ones
        # resample_label_1.loc[:,'text'] = resample_label_0_txt['text']
        # # Set label to 0 since they should not be considered toxic anymore
        # resample_label_1.loc[:,'label'] = 0
        # #Concat to main data
        # train_data_df = pd.concat([resample_label_1,train_data_df])
        # new_suffix += '_text_augmented'
        # print("Augment data by changing text of toxic memes - Option A")



        # Add new upsamples list to main data
        save_new_confounders_data = pd.concat(
            [rows_confounders_upsampled, train_data_df])
        # Shuffle the concatenated data and reset index
        save_new_confounders_data = save_new_confounders_data.sample(
            frac=1, random_state=SEED).reset_index(drop=True)

        # Set the new filename to save upsampled data
        new_suffix += '_upsampled_confounders_'+str(multiplier)+'x_times.jsonl'
        save_new_data = self.filepath.replace('.jsonl', new_suffix)
        save_new_confounders_data.to_json(
            save_new_data, lines=True, orient='records')
        print("Saved confounder samples to: ")
        print(save_new_data)
        return save_new_data

    def _prepare_data_list(self):
        # Check filepath
        assert self.filepath.endswith(
            ".jsonl"), "The filepath requires a JSON list file (\".jsonl\"). Please correct the given filepath \"%s\"" % self.filepath
        self.basepath = self.filepath.rsplit("/", 1)[0]

        if self.upsample_multiplier > 0:
            self.filepath = self.upsample_confounders()

        # YOUR CODE HERE:  Load jsonl file as list of JSON objects stored in 'self.json_list
        with open(self.filepath) as f:
            self.json_list = [json.loads(json_dict)
                              for json_dict in f.readlines()]
        print("Loaded dataset contains ", str(len(self.json_list)), "samples")
        self._load_dataset()
    

    def _load_dataset(self):
        # Loading json files into namespace object
        # Note that if labels do not exist, they are replaced with -1
        self.data = SimpleNamespace(
            ids=None, imgs=None, labels=None, text=None)

        # YOUR CODE HERE:  load the object lists from self.json_list
        self.data.ids = [example["id"] for example in self.json_list]
        self.data.return_ids = True
        # if label doesn't exist, use -1 as default
        self.data.labels = [example.get("label", -1)
                            for example in self.json_list]
        self.data.text = [example["text"] for example in self.json_list]
        self.data.imgs = [example["img"] for example in self.json_list]

        # YOUR CODE HERE:  Check if all image features' and image features' info files exist
        for img_id in self.data.ids:
            img_id = self._expand_id(img_id)

            np_file = os.path.join(self.feature_dir, img_id + ".npy")
            assert os.path.exists(np_file), f"File '{np_file}' does not exist"

            info_np_file = os.path.join(self.feature_dir, img_id + "_info.npy")
            assert os.path.exists(
                info_np_file), f"File '{info_np_file}' does not exist"

        # YOUR CODE HERE:  Iterate over data ids and load img_feats and img_pos_feats into lists (defined above) using _load_img_feature
        both_img_feats = [(self._load_img_feature(
            img_id, normalize=wandb.config.no_normalize_img)) for img_id in self.data.ids]
        # FIXME something might be wrong here
        # split a list of tuples into two separate lists
        self.data.img_feats, self.data.img_pos_feats = zip(*both_img_feats)

        # Preprocess text if selected
        if self.text_preprocess is not None:
            self.data.text = self.text_preprocess(self.data.text)

    def __len__(self):
        # YOUR CODE HERE:  mandatory.
        return len(self.data.ids)

    def _expand_id(self, img_id):
        # YOUR CODE HERE:  Add trailing zeros to the given id (check file names) using zfill
        return str(img_id).zfill(5)

    def _load_img_feature(self, img_id, normalize=True):

        img_id = self._expand_id(img_id)
        # YOUR CODE HERE:  Load image features and image feats info in 'img_feat' and 'img_feat_info' (i.e., .npy and _info.npy files) using _load_img_feature
        img_feat = np.load(os.path.join(self.feature_dir, img_id + ".npy"))
        # the loaded *_info.npy file is a 0-d array, so we use item() to retrieve the dictionary
        img_feat_info = np.load(os.path.join(
            self.feature_dir, img_id + "_info.npy"), allow_pickle=True).item()

        # YOUR CODE HERE:  get the x and y coordinates from 'img_feat_info['bbox']'

        if self.filter_text:  # remove bounding boxes containing text
            object_ids = img_feat_info["objects_id"]

            # get indices of text bounding boxes, the object id of text is 1179
            text_obj_ids = [i for i, obj_id in enumerate(
                object_ids) if obj_id == 1179]

            # remove text bounding boxes
            mask = np.ones(object_ids.size, dtype=bool)
            mask[text_obj_ids] = False
            coords = img_feat_info["bbox"][mask, :]
            img_feat = img_feat[mask, :]
        else:
            # retrieve a matrix where each row i represents [x1, y1, x2, y2] coords (I suppose) of the ith object from the img
            coords = img_feat_info["bbox"]

        if normalize:
            # YOUR CODE HERE:  normalize the coordinates with image width and height
            coords[:, [0, 2]] = coords[:, [0, 2]] / \
                img_feat_info["image_width"]
            coords[:, [1, 3]] = coords[:, [1, 3]] / \
                img_feat_info["image_height"]

        # YOUR CODE HERE:  calculate the width and height of the bbs from their x,y coordinates

        # YOUR CODE HERE:  prepare the 'img_pos_feat' as a 7-dim tensor of x1, y1, x2, y2, w, h, w*h
        img_pos_feat = np.zeros((coords.shape[0], 7))
        # expand coords with width and height
        img_pos_feat[:, :4] = coords
        # compute w = x2 - x1
        img_pos_feat[:, 4] = coords[:, 2] - coords[:, 0]
        # compute h = y2 - y1
        img_pos_feat[:, 5] = coords[:, 3] - coords[:, 1]
        # compute w*h
        img_pos_feat[:, 6] = img_pos_feat[:, 4] * img_pos_feat[:, 5]

        return torch.from_numpy(img_feat).float(), torch.from_numpy(img_pos_feat).float()

    def __getitem__(self, idx):
        # YOUR CODE HERE:  write the return of one item of the batch containing the elements denoted in the return statement
        # HINT: use _load_img_feature

        data_id = self.data.ids[idx]
        label = self.data.labels[idx]
        text = self.data.text[idx]

        img_feat = self.data.img_feats[idx]
        img_pos_feat = self.data.img_pos_feats[idx]

        return {
            'img_feat': img_feat,
            'img_pos_feat': img_pos_feat,
            'text': text,
            'label': label,
            'data_id': data_id}

    def get_collate_fn(self):
        """
        Returns functions to use in the Data loader (collate_fn).
        Image features and position features are stacked (with padding) and returned.
        For text, the function "text_padding" takes all text elements, and is expected to return a list or stacked tensor.
        """

        def collate_fn(samples):
            # samples is a list of dictionaries of the form returned by __get_item__()
            # YOUR CODE HERE: Create separate lists for each element by unpacking
            data_ids, labels, texts, img_feats, img_pos_feats = zip(*[(item["data_id"], item["label"], item["text"], item["img_feat"], item["img_pos_feat"])
                                                                      for item in samples])

            # YOUR CODE HERE:  Pad 'img_feat' and 'img_pos_feat' tensors using pad_sequence
            img_feat = pad_sequence(
                img_feats, batch_first=True, padding_value=0)
            img_pos_feat = pad_sequence(
                img_pos_feats, batch_first=True, padding_value=0)

            # Tokenize and pad text
            if self.text_padding is not None:
                texts = self.text_padding(texts)

            # YOUR CODE HERE:  Stack labels and data_ids into tensors (list --> tensor)
            data_ids = torch.Tensor(list(data_ids))
            labels = torch.Tensor(list(labels))

            # Text input
            input_ids = texts['input_ids']
            text_len = texts['length'].tolist()
            token_type_ids = texts['token_type_ids'] if 'token_type_ids' in texts else None
            position_ids = torch.arange(0, input_ids.shape[1], device=input_ids.device).unsqueeze(
                0).repeat(input_ids.shape[0], 1)

            # Attention mask
            if self.compact_batch:
                img_len = [i.size(0) for i in img_feat]
                attn_mask = get_attention_mask(text_len, img_len)
            else:
                text_mask = texts['attention_mask']
                img_len = [i.size(0) for i in img_feat]
                zero_text_len = [0] * len(text_len)
                img_mask = get_attention_mask(zero_text_len, img_len)
                attn_mask = torch.cat((text_mask, img_mask), dim=1)

            # Gather index
            out_size = attn_mask.shape[1]
            batch_size = attn_mask.shape[0]
            max_text_len = input_ids.shape[1]
            gather_index = get_gather_index(
                text_len, img_len, batch_size, max_text_len, out_size)

            batch = {'input_ids': input_ids,
                     'position_ids': position_ids,
                     'img_feat': img_feat,
                     'img_pos_feat': img_pos_feat,
                     'token_type_ids': token_type_ids,
                     'attn_mask': attn_mask,
                     'gather_index': gather_index,
                     'labels': labels,
                     'ids': data_ids}

            return batch
        return collate_fn


## Use the code below to check your implementation ##
if __name__ == '__main__':
    import argparse
    from functools import partial
    from transformers import BertTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str,
                        help="Path to jsonl file of dataset", required=True)
    parser.add_argument('--feature_dir', type=str,
                        help='Directory containing image features', required=True)
    parser.add_argument(
        '--filter_text', help='Filter out bounding boxes around text', action='store_true')
    args = parser.parse_args()

    # Tokenize
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    tokenizer_func = partial(tokenizer, max_length=256, padding='max_length',
                             truncation=True, return_tensors='pt', return_length=True)
    dataset = MemeDataset(filepath=args.filepath,
                          feature_dir=args.feature_dir,
                          text_padding=tokenizer_func,
                          confidence_threshold=0.4,
                          filter_text=args.filter_text)

    data_loader = data.DataLoader(
        dataset, batch_size=32, collate_fn=dataset.get_collate_fn(), sampler=None)
    logger.info("Length of data loader: %i" % len(data_loader))
    try:
        out_dict = next(iter(data_loader))
        logger.info("Data loading has been successful.")
    except NotImplementedError as e:
        logger.error(
            "Error occured during data loading, please have a look at this:\n" + str(e))
    print("Image features", out_dict['img_feat'].shape)
