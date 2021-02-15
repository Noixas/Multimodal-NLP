from tokenize import Double
import torch
import torch.utils.data as data
import numpy as np
import os
from PIL import Image
from types import SimpleNamespace
import logging
import matplotlib.pyplot as plt
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
                 filepath : str,
                 feature_dir: str = None,
                 text_preprocess = None,
                 text_padding = None,
                 compact_batch: bool = True,
                 confidence_threshold : float = 0.0):
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
        self._prepare_data_list()

    
    
    def _prepare_data_list(self):
        # Check filepath
        assert self.filepath.endswith(".jsonl"), "The filepath requires a JSON list file (\".jsonl\"). Please correct the given filepath \"%s\"" % self.filepath
        self.basepath = self.filepath.rsplit("/",1)[0]
        # YOUR CODE HERE:  Load jsonl file as list of JSON objects stored in 'self.json_list
        with open(self.filepath) as f:
            self.json_list = [json.loads(json_dict) for json_dict in f.readlines()]
        self._load_dataset()

    

    def _load_dataset(self):        
        # Loading json files into namespace object
        # Note that if labels do not exist, they are replaced with -1        
        self.data = SimpleNamespace(ids=None, imgs=None, labels=None, text=None)

        # YOUR CODE HERE:  load the object lists from self.json_list
        self.data.ids = [example["id"] for example in self.json_list]
        self.data.return_ids = True 
        self.data.labels = [example.get("label", -1) for example in self.json_list] # if label doesn't exist, use -1 as default
        self.data.text = [example["text"] for example in self.json_list]
        self.data.imgs = [example["img"] for example in self.json_list]

        # YOUR CODE HERE:  Check if all image features' and image features' info files exist
        # Honestly, I don't exactly understand what Shaan expected from us here.
        for img_id in self.data.ids:
            img_id = self._expand_id(img_id) 

            np_file = os.path.join(self.feature_dir, img_id + ".npy")
            assert os.path.exists(np_file), f"File '{np_file}' does not exist"

            info_np_file = os.path.join(self.feature_dir, img_id + "_info.npy")
            assert os.path.exists(info_np_file), f"File '{info_np_file}' does not exist"
            
        # YOUR CODE HERE:  Iterate over data ids and load img_feats and img_pos_feats into lists (defined above) using _load_img_feature
        both_img_feats = [(self._load_img_feature(img_id)) for img_id in self.data.ids]
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


    def _load_img_feature(self, img_id, normalize=False):
        
        img_id = self._expand_id(img_id)
        # YOUR CODE HERE:  Load image features and image feats info in 'img_feat' and 'img_feat_info' (i.e., .npy and _info.npy files) using _load_img_feature
        img_feat = np.load(os.path.join(self.feature_dir, img_id + ".npy"))
        # the loaded *_info.npy file is a 0-d array, so we use item() to retrieve the dictionary
        img_feat_info = np.load(os.path.join(self.feature_dir, img_id + "_info.npy"), allow_pickle=True).item()
        
        # YOUR CODE HERE:  get the x and y coordinates from 'img_feat_info['bbox']'
        
        # retrieve a matrix where each row i represents [x1, y1, x2, y2] coords (I suppose) of the ith object from the img
        coords = img_feat_info["bbox"]
        
        
        if normalize:
            # YOUR CODE HERE:  normalize the coordinates with image width and height
            coords[:,[0, 2]] = coords[:,[0, 2]] / img_feat_info["image_width"]
            coords[:,[1, 3]] = coords[:,[1, 3]] / img_feat_info["image_height"]
        
        # YOUR CODE HERE:  calculate the width and height of the bbs from their x,y coordinates 

        # YOUR CODE HERE:  prepare the 'img_pos_feat' as a 7-dim tensor of x1, y1, x2, y2, w, h, w*h
        img_pos_feat = np.zeros((coords.shape[0], 7))
        # expand coords with width and height
        img_pos_feat[:,:4] = coords
        # compute w = x2 - x1
        img_pos_feat[:, 4] = coords[:,2] - coords[:,0]
        # compute h = y2 - y1
        img_pos_feat[:, 5] = coords[:,3] - coords[:,1]
        # compute w*h
        img_pos_feat[:, 6] = img_pos_feat[:, 4] * img_pos_feat[:, 5] 
        # FIXME what should be the type of img_feat and img_pos_feat? ndarray? torch tensor?

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
            img_feat = pad_sequence(img_feats, batch_first=True, padding_value=0)
            img_pos_feat = pad_sequence(img_pos_feats, batch_first=True, padding_value=0)

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
            position_ids = torch.arange(0, input_ids.shape[1], device=input_ids.device).unsqueeze(0).repeat(input_ids.shape[0], 1)

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
            gather_index = get_gather_index(text_len, img_len, batch_size, max_text_len, out_size)
            
            batch = {'input_ids': input_ids,
                    'position_ids': position_ids,
                    'img_feat': img_feat,
                    'img_pos_feat': img_pos_feat,
                    'token_type_ids': token_type_ids,
                    'attn_mask': attn_mask,
                    'gather_index': gather_index,
                    'labels': labels,
                    'ids' : data_ids}
            
            return batch        
        return collate_fn
    



## Use the code below to check your implementation ##

if __name__ == '__main__':
    import argparse
    from functools import partial
    from transformers import BertTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, help="Path to jsonl file of dataset", required=True)
    parser.add_argument('--feature_dir', type=str, help='Directory containing image features', required=True)
    args = parser.parse_args()

    # Tokenize
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    tokenizer_func = partial(tokenizer, max_length=256, padding='max_length',
                             truncation=True, return_tensors='pt', return_length=True)
    dataset = MemeDataset(filepath=args.filepath,
                          feature_dir=args.feature_dir,
                          text_padding=tokenizer_func,
                          confidence_threshold=0.4)
    data_loader = data.DataLoader(dataset, batch_size=32, collate_fn=dataset.get_collate_fn(), sampler=None)
    logger.info("Length of data loader: %i" % len(data_loader))
    try:
        out_dict = next(iter(data_loader))
        logger.info("Data loading has been successful.")
    except NotImplementedError as e:
        logger.error("Error occured during data loading, please have a look at this:\n" + str(e))
    print("Image features", out_dict['img_feat'].shape)