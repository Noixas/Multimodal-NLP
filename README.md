# NLP-2-Assignment-Multimodal-NLP
This is the official repository for the Assignment on Multimodal NLP for the MS in AI course NLP-2 at University of Amsterdam.

### Installation

- Create a virtual environment with Python 3.7.5 using either `virtualenv` or `conda`.
- Activate the virtual environment.
- Install the required packages using `pip install -r requirements.txt`. 
- Install pytorch 1.6.0 with Cuda 10.1 using `conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch` or `pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html`
- Install Nvidia Apex as follows:
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

**NOTE:** In case you want to work on Google Colab, we also provide a minimal working notebook (`NLP-2-Assignment-Multimodal-NLP.ipynb`) for the same. Please refer to it for further instructions. 


### Clone the repository

Clone the repository using `git clone https://github.com/shaanchandra/NLP-2-Assignment-Multimodal-NLP.git`.

### Download the pretrained model

- Navigate to the directory: `cd NLP-2-Assignment-Multimodal-NLP/model_checkpoints`.
- The pretrained UNITER-base model can be downloaded using `wget 'https://convaisharables.blob.core.windows.net/uniter/pretrained/uniter-base.pt'`.
- Next, convert the model's state_dict to work with the code using the following snippet:
```python
import torch
model_name = 'uniter-base.pt'
checkpoint = torch.load(model_name)
state_dict = {'model_state_dict': checkpoint}
torch.save(state_dict, model_name)

```

### Obtain image features

- Make a new directory: `mkdir dataset`.
- Copy the HatefulMemes dataset to the `dataset` directory.
- We provide the extracted features [here](https://drive.google.com/file/d/1vTl31tkkm_kpOsL7f3rhGWQFke2y96g_/view?usp=sharing).

### Training

The directory structure is assumed to be as follows:
<pre>
.
├── NLP-2-Assignment-Multimodal-NLP/
├── dataset
│   ├── img/
│   ├── own_features/
│   ├── train.jsonl
│   ├── dev_seen.jsonl
│   ├── dev_unseen.jsonl
│   ├── test_seen.jsonl
│   ├── test_unseen.jsonl
</pre>

To train the model from inside the aforementioned directory structure, run the following:
```bash
python -u train_uniter.py --config config/uniter-base.json --data_path ./dataset --model_path ./model_checkpoints --pretrained_model_file uniter-base.pt --feature_path ./dataset/own_features --lr 3e-5 --scheduler warmup_cosine --warmup_steps 500 --max_epoch 30 --batch_size 16 --patience 5 --gradient_accumulation 2 --model_save_name meme.pt --seed 43 
```
The results will be exported as CSV files in the `model_checkpoints` directory in the format expected to be submitted for the leaderboard.
