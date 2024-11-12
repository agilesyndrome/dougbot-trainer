
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import TrainingArguments, DataCollatorWithPadding, Trainer
import wandb

class LLM:

    model = None
    model_name = None
    tokenizer = None

    def __str__(self):
        model_str = [self.model_name]

        if self.save_as:
            model_str.append(f'--> {self.save_as}:private')

        if self.use_fp16:
            model_str.append('fp16:true')

        model_str.append(f'epochs:{self.training_epochs}')
        model_str.append(f'max_token_length:{self.max_tokenizer_length}' )
        return ' '.join(model_str)




    def __init__(self, config):

        self.model_name = config.DOUGBOT_BASE_MODEL
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.max_tokenizer_length = self.tokenizer.model_max_length
        self.truncate_tokens = True

        self.use_fp16 = config.use_fp16


        self.save_as = config.HF_MODEL_OUTPUT

        self.wandb_entity = config.WANDB_ENTITY
        self.wandb_project = config.WANDB_PROJECT
        self.wandb_name = config.WANDB_NAME






        self.training_epochs = config.training_epochs

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model.to(self.device)




    # Step 4: Fine-tune Model
    def train(self, ds):

        if self.wandb_project:
            wandb.init(
                entity=self.wandb_entity,
                project=self.wandb_project,
                name=self.__str__()
            )

        #collator = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="pt", padding=True)



        train_ds = ds.ds

        # The model did not return a loss from the inputs, only the following keys: logits. For reference, the inputs it received are input_ids,attention_mask.
        #tokens['train']['labels'] = tokens['train']['input_ids']

        training_args = TrainingArguments(
            output_dir="./output",
            per_device_train_batch_size=1,

            # Saves memory by accumulating gradients
            gradient_accumulation_steps=3,

            # unless your model is very small (gpt2-ish), you should use gradient checkpointing
            # so we don't have to load the entire model into memory all the time
            gradient_checkpointing=True,
            num_train_epochs=self.training_epochs,
            logging_dir="./logs",
            logging_steps=10,
            save_steps=10_000,
            save_total_limit=2,
            run_name=f'dougbot-{self.model_name.replace("/", "_")}',
            fp16=self.use_fp16
        )

        # the importance of data_collator is to pad the data to the same length




        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            # eval_dataset=tokens['validation'],
            # data_collator=collator

        )

        trainer.train()



        trainer.save_model(f'./dougbot_{self.model_name.replace("/", "_")}')
        self.tokenizer.save_pretrained(f'./dougbot_{self.model_name.replace("/", "_")}', set_lower_case=False)


        if save_as:
            self.model.save_to_hub(self.save_as, token=os.getenv('HUGGINGFACE_TOKEN'), privat=True)
            self.tokenizer.save_to_hub(self.save_as, token=os.getenv('HUGGINGFACE_TOKEN'), private=True)


