import torch
import torch.utils.data as data
import numpy as np
import os
from PIL import Image
import logging
import matplotlib.pyplot as plt


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

        self._prepare_data_list()

    
    
    def _prepare_data_list(self):
        # Check filepath
        assert self.filepath.endswith(".jsonl"), "The filepath requires a JSON list file (\".jsonl\"). Please correct the given filepath \"%s\"" % self.filepath
        self.basepath = self.filepath.rsplit("/",1)[0]
        # YOUR CODE HERE:  Load jsonl file as list of JSON objects stored in 'self.json_list
        
        self._load_dataset()

    

    def _load_dataset(self):        
        # Loading json files into namespace object
        # Note that if labels do not exist, they are replaced with -1        
        self.data = SimpleNamespace(ids=None, imgs=None, labels=None, text=None)

        # YOUR CODE HERE:  load the object lists from self.json_list
        self.data.ids = # fill me
        self.data.labels = # fill me
        self.data.text = # fill me
        self.data.imgs = # fill me

        # YOUR CODE HERE:  Check if all image features' and image features' info files exist
        # YOUR CODE HERE:  Iterate over data ids and load img_feats and img_pos_feats into lists (defined above) using _load_img_feature
        
        # Preprocess text if selected
        if self.text_preprocess is not None:
            self.data.text = self.text_preprocess(self.data.text)

    
    def __len__(self):
        # YOUR CODE HERE:  mandatory. 
        raise NotImplementedError


    def _expand_id(self, img_id):
        # YOUR CODE HERE:  Add trailing zeros to the given id (check file names) using zfill
        raise NotImplementedError


    def _load_img_feature(self, img_id, normalize=False):
        img_id = self._expand_id(img_id)
        # YOUR CODE HERE:  Load image features and image feats info in 'img_feat' and 'img_feat_info' (i.e., .npy and _info.npy files) using _load_img_feature

        # YOUR CODE HERE:  get the x and y coordinates from 'img_feat_info['bbox']'
        
        
        if normalize:
            # YOUR CODE HERE:  normalize the coordinates with image width and height

        # YOUR CODE HERE:  calculate the width and height of the bbs from their x,y coordinates 
        
        # YOUR CODE HERE:  prepare the 'img_pos_feat' as a 7-dim tensor of x1, y1, x2, y2, w, h, w*h

        return img_feat, img_pos_feat

    
    
    def __getitem__(self, idx):
        # YOUR CODE HERE:  write the return of one item of the batch conataining the elements denoted in the return statement
        # HINT: use _load_img_feature

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

            # YOUR CODE HERE:  Pad 'img_feat' and 'img_pos_feat' tensors using pad_sequence

            # Tokenize and pad text
            if self.text_padding is not None:
                texts = self.text_padding(texts)
            
            # YOUR CODE HERE:  Stack labels and data_ids into tensors (list --> tensor)

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
    data_loader = data.DataLoader(dataset, batch_size=32, collate_fn=dataset.get_collate_fn(), sampler=ConfounderSampler(dataset, repeat_factor=2))
    logger.info("Length of data loader: %i" % len(data_loader))
    try:
        out_dict = next(iter(data_loader))
        logger.info("Data loading has been successful.")
    except NotImplementedError as e:
        logger.error("Error occured during data loading, please have a look at this:\n" + str(e))
    print("Image features", out_dict['img_feat'].shape)