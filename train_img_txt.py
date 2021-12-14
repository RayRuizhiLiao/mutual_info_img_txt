import os
import argparse
import logging
import json
import numpy as np
import cv2

import torch
from pytorch_transformers import BertTokenizer

from mutual_info_img_txt import model_utils
from mutual_info_img_txt.main_utils import ImageTextModelManager


current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)

parser = argparse.ArgumentParser()

parser.add_argument('--image_dir', type=str,
                    default=os.path.join(current_dir, 'example_data/images/'),
                    help='The image data directory')
parser.add_argument('--text_data_dir', type=str,
                    default=os.path.join(current_dir, 'example_data/text/'),
                    help='The text data directory')
parser.add_argument('--bert_pretrained_dir', type=str,
                    default=os.path.join(current_dir, 'bert_pretrain_all_notes_150000'),
                    help='The directory that contains a pretrained BERT model')
parser.add_argument('--bert_config_name',
                    default='bert_config.json', help='Bert model config file')
parser.add_argument('--save_dir', type=str,
                    default=os.path.join(current_dir, 'save_dir'))
parser.add_argument('--dataset_metadata', type=str,
                    default=os.path.join(current_dir, 'example_data/training_chexpert_mini.csv'),
                    help='The metadata for the model training ')

parser.add_argument('--batch_size', default=8, type=int,
                    help='Mini-batch size')
parser.add_argument('--num_train_epochs', default=100, type=int,
                    help='Number of training epochs')
parser.add_argument('--mi_estimator', type=str,
                    default='infonce',
                    help='Mutual information estimator (variational bound)')
parser.add_argument('--init_lr', default=5e-4, type=float, 
                    help='Intial learning rate')

parser.add_argument('--max_seq_length', default=320, type=int, 
                    help='Maximum sequence length for the BERT model')
parser.add_argument('--img_size', default=256, type=int,
                    help='The size of the input image')
parser.add_argument('--output_channels', default=1, type=int,
                    help='The number of ouput channels for the classifier')
parser.add_argument('--image_model_name', default='resnet256_6_2_1', type=str,
                    help='Neural network architecture to be used for image model')

def train_image_text():
    args = parser.parse_args()

    print(f"Initial args: {args}")
    
    '''
    Check cuda
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert torch.cuda.is_available(), "No GPU/CUDA is detected!"

    '''
    Create a sub-directory under save_dir 
    '''
    args.save_dir = os.path.join(args.save_dir, 
                                 f'{args.mi_estimator}_total_epochs{args.num_train_epochs}')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    '''
    Configure the log file
    '''
    log_path = os.path.join(args.save_dir, 'training.log')
    logging.basicConfig(filename=log_path, level=logging.INFO, filemode='w', 
    					format='%(asctime)s - %(name)s %(message)s', 
    					datefmt='%m-%d %H:%M')

    logger = logging.getLogger(__name__)
    logger.info(f"args: {args}")

    print(f"Updated args: {args}")

    '''
    Tokenize text
    '''
    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrained_dir)
    text_token_features = model_utils.load_and_cache_examples(args, tokenizer)

    '''
    Initialize a joint image-text model manager
    '''
    model_manager = ImageTextModelManager(bert_pretrained_dir=args.bert_pretrained_dir, 
                                    	  bert_config_name=args.bert_config_name,
                                    	  output_channels=args.output_channels,
                                    	  image_model_name=args.image_model_name)

    '''
    Train the joint model
    '''
    model_manager.train(text_token_features=text_token_features, 
                        device=device,
                        args=args)

train_image_text()
