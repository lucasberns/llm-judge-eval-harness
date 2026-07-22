import numpy as np
import random

from datasets import load_dataset

def download_data():
    DATA = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base")

def create_csv():
    for i in range(300):
        CHOSEN = DATA['chosen'][i]
        REJECTED = DATA['rejected'][i]

        