import os

from datasets.utils.py_utils import classproperty

DEFAULT_BASE_MODEL = 'openai-community/gpt2'
DEFAULT_TRAINING_EPOCHS = 3

class Config:

    def __init__(self):
        self.HF_TOKEN = os.getenv("HF_TOKEN", None) # Hugging Face API token


        # What dataset will we load to begin our training?
        self.HF_DATASET = os.getenv("HF_DATASET", None) # Hugging Face dataset name

        self.DOUGBOT_BASE_MODEL = os.getenv("DOUGBOT_BASE_MODEL", DEFAULT_BASE_MODEL) # Hugging Face model name

        self.HF_MODEL_OUTPUT = os.getenv("HF_MODEL_OUTPUT", None) # Hugging Face model output path


        self.WANDB_PROJECT = os.getenv("WANDB_PROJECT", None) # Weights and Biases project name
        self.WANDB_ENTITY = os.getenv("WANDB_ENTITY", None) # Weights and Biases entity name
        self.WANDB_NAME = os.getenv("WANDB_NAME", None) # Weights and Biases run name

    @property
    def training_epochs(self):
        training_epochs = DEFAULT_TRAINING_EPOCHS
        try:
            training_epochs = int(os.getenv('DOUGBOT_TRAINING_EPOCHS', DEFAULT_TRAINING_EPOCHS))
        except:
        # who the fuck cares, just use 3 epochs
            pass

        return training_epochs



    @property
    def use_fp16(self):
        fp16 = os.getenv('DOUGBOT_USEFP16', False)

        if type(fp16) == str:
            match fp16.lower():
                case 'true':
                    fp16 = True
                case 'false':
                    fp16 = False
                case '1':
                    fp16 = True
                case '0':
                    fp16 = False
                case 'yes':
                    fp16 = True
                case 'no':
                    fp16 = False
                case _:
                    raise Exception(f'Invalid value {fp16} for DOUGBOT_USEFP16')

        return fp16

