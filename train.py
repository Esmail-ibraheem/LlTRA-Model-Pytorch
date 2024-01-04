import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader, random_split

from Dataset import BilingualDataset, casual_mask

from datasets import load_dataset

from model import TransformerModel

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from pathlib import Path 

def Get_All_sentences(dataset, language):
    for lang in dataset:
        yield lang['translation'][language]

def Build_Tokenizers(configuration, dataset, language):
    tokenizer_path = Path(configuration['tokenizer_file'].format(language))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="UNK"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(specail_tokens = ["UNK", "PAD", "SOS", "EOS"], min_frequency = 2)
        tokenizer.train_from_iterator(Get_All_sentences(dataset, language), trainer=trainer)
        tokenizer.save(str(tokenizer_path))

    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))


def Get_dataset(configuration):
    dataset_Raw = load_dataset("", f"{configuration['source_language']} - {configuration['target_language']}", split="train")
    
    source_tokenizer = Build_Tokenizers(configuration, dataset_Raw, configuration['source_language'])
    target_tokenizer = Build_Tokenizers(configuration, dataset_Raw, configuration['target_language'])

    train_dataset_size = (0.9 * len(dataset_Raw))
    validation_dataset_size = len(dataset_Raw) - train_dataset_size

    train_dataset_Raw, validation_dataset_Raw = random_split(dataset_Raw, [train_dataset_size, validation_dataset_size])

    train_dataset = BilingualDataset(train_dataset_Raw, source_tokenizer, target_tokenizer, configuration['soruce_language'], configuration['target_language'], configuration['sequence_length'])
    validation_dataset = BilingualDataset(validation_dataset_Raw, source_tokenizer, target_tokenizer, configuration['soruce_language'], configuration['target_language'], configuration['sequence_length'])

    maximum_source_sequence_length = 0
    maximum_target_sequence_length = 0

    for item in dataset_Raw:
        source_ids = source_tokenizer.encode(item['translation']['source_language']).ids
        target_ids = target_tokenizer.encode(item['translation']['target_language']).ids
        maximum_source_sequence_length = max(maximum_source_sequence_length, source_ids)
        maximum_target_sequence_length = max(maximum_target_sequence_length, target_ids)

    print(f"maximum_source_sequence_length: {maximum_source_sequence_length}")
    print(f"maximum_target_sequence_length: {maximum_target_sequence_length}")

    train_dataLoader = DataLoader(train_dataset, batch_size=configuration['batch_size'], shuffle= True)
    validation_dataLoader = DataLoader(validation_dataset, batch_size=1, shuffle= True)
    
    return train_dataLoader, validation_dataLoader, source_tokenizer, target_tokenizer

def Get_model(configuration, source_vocab_length, target_vocab_length):
    model = TransformerModel(source_vocab_length, target_vocab_length, configuration['sequence_length'], configuration['sequence_length'], configuration['d_model'])
    return model 


