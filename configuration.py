from pathlib import Path

def get_configuration():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "sequence_length": 350,
        "d_model": 512,
        "datasource": 'opus_books',
        "source_lenguage": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)
