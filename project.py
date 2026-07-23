import numpy as np
import random
import pandas as pd
import nltk
from difflib import SequenceMatcher

from datasets import load_dataset

def download_data():
    DATA = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base")
    return DATA

def compare_phrases(PHRASE_1, PHRASE_2):
    MATCHER = SequenceMatcher(None, PHRASE_1, PHRASE_2)
    RATIO = MATCHER.ratio()

    if (MATCHER >= 0.8):
        return "A"
    else:
        return "B"

def create_csv(DATA):
    ANSWER_A = []
    ANSWER_B = []
    PROMPT = []
    CHOSEN = []

    for i in range(300):
        HUMAN_CHOSEN = DATA['train']['chosen'][i]
        HUMAN_REJECTED = DATA['train']['rejected'][i]

        PARSED_CHOSEN = [item.strip() for item in HUMAN_CHOSEN.split("\n")]
        PARSED_REJECTED = [item.strip() for item in HUMAN_REJECTED.split("\n")]

        FINAL_A = PARSED_CHOSEN[-1]
        FINAL_B = PARSED_REJECTED[-1]

        START = " ".join(PARSED_CHOSEN[:-1])

        ANSWERS = [FINAL_A, FINAL_B]
        random.shuffle(ANSWERS)
        ANS_A, ANS_B = ANSWERS

        CORRECT = compare_phrases(ANS_A, FINAL_A)

        ANSWER_A.append(ANS_A)
        ANSWER_B.append(ANS_B)
        PROMPT.append(START)
        CHOSEN.append(CORRECT)

    CREATOR = {"PROMPT": PROMPT, "ANSWER_A": ANSWER_A, "ANSWER_B": ANSWER_B, "CORRECT": CHOSEN}
    pd.DataFrame(data=CREATOR).to_csv('treated_data.csv', index=False)


def main():
    DATA = download_data()
    create_csv(DATA)

if __name__ == "__main__":
    main()