from importlib.resources import path
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from datasets import load_dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from pathlib import Path

def Get_All_Sentences(dataset, language):
    for lang in dataset:
        yield lang['translation'][language]

def Build_Tokenizer(configuration, dataset, language):
    tokenizer_path = Path(configuration["tokenizer_file"].format(language))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="UNK"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ["UNK", "SOS", "EOS"], min_frequency = 2)
        tokenizer.train_from_iterator(Get_All_Sentences(dataset, language) ,trainer = trainer)
        tokenizer.save(str(tokenizer_path))

    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

def Load_Dataset(configuration, dataset, language):
    dataset_Raw = load_dataset(" ", f"{configuration['source_language']} - {configuration['target_language']}", split="train")

    source_dataset_Raw = Build_Tokenizer(configuration, dataset_Raw, configuration['source_language'])
    target_dataset_Raw = Build_Tokenizer(configuration, dataset_Raw, configuration['target_language'])

    train_dataset_size = (0.9 * len(dataset_Raw))
    validation_dataset_size = (len(dataset_Raw) - train_dataset_size)

    train_dataset_Raw, validation_dataset_Raw = random_split(dataset_Raw, [train_dataset_size, validation_dataset_size])

    