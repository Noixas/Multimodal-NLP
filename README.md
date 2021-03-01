# NLP-2-Assignment-Multimodal-NLP
This is a modified version from the original README.md, please refer to README_ORIGINAL.md to access that file.

The code pipeline has been streamlined and after following the installation below, running `main.ipynb` should be enough to replicate the results of the paper. 

### Installation
Here we outline the steps required to run `main.ipynb`. These steps were used in a machine with Ubuntu 20.04 LTS and a GPU RTX 2060 Super.

- Install Anaconda in the machine.
- Create a virtual environment with Python 3.7.5 using `conda`. E.g. Run in the terminal `conda create --name "nlp2-multimodal-R-B python=3.7.5"`.
- Activate environment with `conda activate nlp2-multimodal-R-B `
- Clone the repository in the desired filepath. `git clone https://github.com/Noixas/Multimodal-NLP.git`

- To access the data, register in the [Hateful Memes challenge](https://www.drivendata.org/competitions/64/hateful-memes/data/)
    - Download the data and extract the zip file in the folder 'dataset'.
- Folder structure should look as follows:
<pre>
.
├── Multimodal-NLP/
│       ├── dataset
│       ├── img/
│       ├── own_features/
│       ├── train.jsonl
│       ├── dev_seen.jsonl
│       ├── dev_unseen.jsonl
│       ├── test_seen.jsonl
│       ├── test_unseen.jsonl
</pre>

- In the terminal go to the path where the repository was cloned. E.g. `/home/username/Documents/Multimodal-NLP/`
- Run `jupyter notebook` in the terminal to start a session and open the jupyter tree, it will show all the files in the current folder.
- Click on `main.ipynb` and run all the cells in the notebook, it will install the python libraries that are required and download the rest of the data that is needed. 
    - If you face problems running the installation part in the notebook, try using the commands directly on the terminal or leave an issue in the repository.


To train the model from the root folder of the repository, run the following in the terminal:
```bash
python -u train_uniter.py --config config/uniter-base.json --data_path ./dataset --model_path ./model_checkpoints --pretrained_model_file uniter-base.pt --feature_path ./dataset/own_features --lr 3e-5 --scheduler warmup_cosine --warmup_steps 500 --max_epoch 30 --batch_size 16 --patience 5 --gradient_accumulation 2 --model_save_name meme.pt --seed 43 
```