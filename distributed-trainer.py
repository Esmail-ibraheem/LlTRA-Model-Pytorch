import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# Distributed training
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group

import warnings
from tqdm import tqdm
import os
from pathlib import Path
import argparse

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import wandb
import torchmetrics

from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_default_config, get_weights_file_path, get_latest_weights_file_path, ModelConfig

def greedy_decode(model: nn.Module, source: torch.Tensor, source_mask: torch.Tensor, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer, max_len: int, device: torch.device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.module.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.module.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.module.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model: nn.Module, validation_ds: DataLoader, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer, max_len: int, device: torch.device, print_msg: callable, global_step: int, num_examples: int = 2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) 
            encoder_mask = batch["encoder_mask"].to(device) 

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    
    metric = torchmetrics.CharErrorRate()
    cer = metric(predicted, expected)
    wandb.log({'validation/cer': cer, 'global_step': global_step})

    # Compute the word error rate
    metric = torchmetrics.WordErrorRate()
    wer = metric(predicted, expected)
    wandb.log({'validation/wer': wer, 'global_step': global_step})

    # Compute the BLEU metric
    metric = torchmetrics.BLEUScore()
    bleu = metric(predicted, expected)
    wandb.log({'validation/BLEU': bleu, 'global_step': global_step})

def get_all_sentences(ds: Dataset, lang: str):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config: ModelConfig, ds: Dataset, lang: str) -> Tokenizer:
    tokenizer_path = Path(config.tokenizer_file.format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config: ModelConfig):
    # It only has the train split, so we divide it overselves
    ds_raw = load_dataset('opus_books', f"{config.lang_src}-{config.lang_tgt}", split='train')

    # Build tokenizers
    print(f"GPU {config.local_rank} - Loading tokenizers...")
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config.lang_src)
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config.lang_tgt)

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config.lang_src, config.lang_tgt, config.seq_len)
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config.lang_src, config.lang_tgt, config.seq_len)

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config.lang_src]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config.lang_tgt]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'GPU {config.local_rank} - Max length of source sentence: {max_len_src}')
    print(f'GPU {config.local_rank} - Max length of target sentence: {max_len_tgt}')
    

    train_dataloader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=False, sampler=DistributedSampler(train_ds, shuffle=True))
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config: ModelConfig, vocab_src_len: int, vocab_tgt_len: int):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config.seq_len, config.seq_len, d_model=config.d_model)
    return model

def train_model(config: ModelConfig):
    # Define the device
    assert torch.cuda.is_available(), "Training on CPU is not supported"
    device = torch.device("cuda")
    print(f"GPU {config.local_rank} - Using device: {device}")

    # Make sure the weights folder exists
    Path(config.model_folder).mkdir(parents=True, exist_ok=True)

    # Load the dataset
    print(f"GPU {config.local_rank} - Loading dataset...")
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-9)

    # By default, load the latest checkpoint
    initial_epoch = 0
    global_step = 0
    wandb_run_id = None
    if config.preload != '':

        if config.preload == 'latest':
            # Get the filename of the latest checkpoint
            model_filename = get_latest_weights_file_path(config)
        else:
            # In case we want to preload a specific checkpoint
            model_filename = get_weights_file_path(config, int(config.preload))

        if model_filename is not None:
            print(f'GPU {config.local_rank} - Preloading model {model_filename}')
            state = torch.load(model_filename)
            model.load_state_dict(state['model_state_dict'])
            initial_epoch = state['epoch'] + 1
            optimizer.load_state_dict(state['optimizer_state_dict'])
            global_step = state['global_step']
            wandb_run_id = state['wandb_run_id']
            del state
        else:
            print(f'GPU {config.local_rank} - Could not find model to preload: {config.preload}. Starting from scratch')

    if config.local_rank == 0:
        wandb.init(
            # set the wandb project where this run will be logged
            project="pytorch-transformer-distributed",
            # allow resuming existing run with the same name (in case the rank 0 node crashed)
            name=f"global_rank_{config.global_rank}",
            id=wandb_run_id,
            resume="allow",
            group=config.wandb_group,
            config=config
        )

    model = DistributedDataParallel(model, device_ids=[config.local_rank])

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    if config.global_rank == 0:
        wandb.define_metric("global_step")
        wandb.define_metric("validation/*", step_metric="global_step")
        wandb.define_metric("train/*", step_metric="global_step")

    for epoch in range(initial_epoch, config.num_epochs):
        torch.cuda.empty_cache()
        model.train()

        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d} on rank {config.global_rank}", disable=config.local_rank != 0)

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            proj_output = model(encoder_input, encoder_mask, decoder_input, decoder_mask)

            label = batch['label'].to(device) # (B, seq_len)

            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}", "global_step": global_step})
        
            if config.local_rank == 0:
                # Log the loss
                wandb.log({'train/loss': loss.item(), 'global_step': global_step})

            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        if config.global_rank == 0:
            # Run validation at the end of every epoch
            run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config.seq_len, device, lambda msg: batch_iterator.write(msg), global_step)

            # Save the model at the end of every epoch
            model_filename = get_weights_file_path(config, epoch)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(), # Need to access module because we are using DDP
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step,
                'wandb_run_id': wandb.run.id # Save to resume logging data
            }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    config = get_default_config()

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=config.batch_size)
    parser.add_argument('--num_epochs', type=int, default=config.num_epochs)
    parser.add_argument('--lr', type=float, default=config.lr)
    parser.add_argument('--seq_len', type=int, default=config.seq_len)
    parser.add_argument('--d_model', type=int, default=config.d_model)
    parser.add_argument('--lang_src', type=str, default=config.lang_src)
    parser.add_argument('--lang_tgt', type=str, default=config.lang_tgt)
    parser.add_argument('--model_folder', type=str, default=config.model_folder)
    parser.add_argument('--model_basename', type=str, default=config.model_basename)
    parser.add_argument('--preload', type=str, default=config.preload)
    parser.add_argument('--tokenizer_file', type=str, default=config.tokenizer_file)
    parser.add_argument('--wandb_group', type=str, default="exp1")
    args = parser.parse_args()

    config.__dict__.update(vars(args))

    config.local_rank = int(os.environ['LOCAL_RANK'])
    config.global_rank = int(os.environ['RANK'])

    assert config.local_rank != -1, "LOCAL_RANK environment variable not set"
    assert config.global_rank != -1, "RANK environment variable not set"

    if config.local_rank == 0:
        print("Configuration:")
        for key, value in config.__dict__.items():
            print(f"{key:>20}: {value}")

    init_process_group(backend='nccl')
    torch.cuda.set_device(config.local_rank)
    
    # Train the model
    train_model(config)

    destroy_process_group()
