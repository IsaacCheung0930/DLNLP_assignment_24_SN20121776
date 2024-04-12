from transformers import AutoTokenizer, DataCollatorWithPadding, TrainerCallback
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, NamedSplit
import evaluate
from copy import deepcopy
import shutil
import numpy as np
import json
import torch

class CustomCallback(TrainerCallback):
    '''
    A class that for customised Huggingface callback.

    Parameter:
        trainer (Trainer):
            The trainer of the Huggingface model
    
    Method:
        on_step_end(args, state, control, **kwargs):
            Triggers at the end of a step for training accuracy and loss. 
    '''
    def __init__(self, trainer) -> None:
        '''
        Initiate the custom callback. 

        Parameter:
            trainer (Trainer):
                The trainer of the Huggingface model
        '''
        super().__init__()
        self._trainer = trainer
    
    def on_step_end(self, args, state, control, **kwargs):
        '''
        Triggers at the end of a step for training accuracy and loss. 

        Parameters:
            args (object): 
                Defined training arguments.
            state (TrainingState): 
                Current training state.
            control (TrainingControl): 
                Training control object.
            **kwargs: 
                Additional arguments.
        
        Return:
            control_copy (object):
                Indicate if training is continued.
        '''
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

class Transformer():
    '''
    A class for transformer model for training, evaluation and predicting. 

    Parameters:
        train_df (DataFrame):
            The dataframe for train set. 
        val_df (DataFrame):
            The dataframe for validation set. 
        test_df (DataFrame):
            The dataframe for test set. 
    
    Methods:
        training(overwrite=True):
            The training process of the fine-tuned transformer.
        evaluation():
            Evaluate the transformer model.
        prediction():
            Predict labels from the fine-tuned model
        get_evaluation_info():
            Return all evalutaion information.
    '''
    def __init__(self, train_df, val_df, test_df):
        '''
        Initiate the transformer model.

        Parameters:
            train_df (DataFrame):
                The dataframe for train set. 
            val_df (DataFrame):
                The dataframe for validation set. 
            test_df (DataFrame):
                The dataframe for test set. 
        '''
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        id2label = {0:'1', 1:'2', 2:'3', 3:'4', 4:'5', 5:'6', 6:'7', 7:'8', 8:'9', 9:'10'}
        label2id = {'1':0, '2':1, '3':2, '4':3, '5':4, '6':5, '7':6, '8':7, '9':8,'10':9}

        model_name = "distilbert/distilbert-base-uncased"
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 10, id2label=id2label, label2id=label2id).to(device)
        
        self._data_collator = DataCollatorWithPadding(tokenizer=self._tokenizer)
        self._accuracy = evaluate.load("accuracy")

        train_df['label'] = [label2id[str(label)] for label in train_df['label']]
        val_df['label'] = [label2id[str(label)] for label in val_df['label']]
        test_df['label'] = [label2id[str(label)] for label in test_df['label']]

        train_dataset, val_dataset, test_dataset = self._custom_dataset(train_df, val_df, test_df)
        self._tokenized_train_dataset = train_dataset.map(self._auto_tokenizer, batched=True)
        self._tokenized_val_dataset = val_dataset.map(self._auto_tokenizer, batched=True)
        self._tokenized_test_dataset = test_dataset.map(self._auto_tokenizer, batched=True)

    def _custom_dataset(self, train_df, val_df, test_df):
        '''
        Reformat the dataframes into Huggingface's Dataset. 

        Parameters:
            train_df (DataFrame):
                The dataframe for train set. 
            val_df (DataFrame):
                The dataframe for validation set. 
            test_df (DataFrame):
                The dataframe for test set. 
        
        Returns:
            train_dataset (Dataset):
                The dataset for train set. 
            val_dataset (Dataset):
                The dataset for validation set. 
            test_dataset (Dataset):
                The dataset for test set. 

        '''
        train_dataset = Dataset.from_pandas(train_df[['label', 'question']], split=NamedSplit('train'), preserve_index=True)
        val_dataset = Dataset.from_pandas(val_df[['label', 'question']], split=NamedSplit('val'), preserve_index=True)
        test_dataset = Dataset.from_pandas(test_df[['label', 'question']], split=NamedSplit('test'), preserve_index=True)

        return train_dataset, val_dataset, test_dataset
    
    def _auto_tokenizer(self, dataset):
        '''
        Tokenise the question column of the dataset. 

        Parameter:
            dataset (Dataset)
                The dataset to be tokenised.

        Return:
            self._tokenizer(dataset['question']):
                The tokenised question column.
        '''
        return self._tokenizer(dataset['question'])
    
    def _evaluation_metrics(self, pred_true):
        '''
        The evaluation metric (accuracy) for the Huggingface transformer.

        Parameter:
            pred_true (tuple):
                The predicted and true labels
        Return:
            self._accuracy.compute(predictions=pred_labels, references=true_labels)
                The prediction accuracy.
        '''
        pred_labels, true_labels = pred_true
        pred_labels = np.argmax(pred_labels, axis=1)
        return self._accuracy.compute(predictions=pred_labels, references=true_labels)
    
    def training(self, overwrite=True):
        '''
        Fine tune the Huggingface trainer.

        Parameter:
            overwrite (bool):
                Determine if the existing outputs are overwritten.
        '''
        training_args = TrainingArguments(
            output_dir="B/results",
            overwrite_output_dir=overwrite,
            logging_dir="B/logs",
            evaluation_strategy="steps",
            logging_strategy="steps",
            save_strategy="steps",
            eval_steps=0.1,
            logging_steps=0.1,
            save_steps=0.1,
            load_best_model_at_end=True,
            num_train_epochs=2,
            per_device_train_batch_size=64,
            per_device_eval_batch_size=64,
            learning_rate=2e-5,
            weight_decay=0.01,
        )   

        self._trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=self._tokenized_train_dataset,
            eval_dataset=self._tokenized_val_dataset,
            tokenizer=self._tokenizer,
            data_collator=self._data_collator,
            compute_metrics=self._evaluation_metrics
        )

        if overwrite:
            shutil.rmtree("B/results")

        self._trainer.add_callback(CustomCallback(self._trainer))
        self._trainer.train()
    
    def evaluation(self):
        '''
        Evaluate the trained model. 
        '''
        valid_results = self._trainer.evaluate(eval_dataset=self._tokenized_test_dataset)
        print("-" * 45)
        print(f"| Test accuracy:  {valid_results['eval_accuracy']:5.3f} | test loss:  {valid_results['eval_loss']:5.3f} |")
        print("-" * 45)

    def prediction(self):
        '''
        Make predictions based on the trained model.

        Return:
            true_values (list):
                True labels.
            pred_values (list):
                Predicted labels. 
        '''
        true_values = np.array(self._tokenized_test_dataset["label"])
        pred = self._trainer.predict(test_dataset=self._tokenized_test_dataset)
        pred_values = np.argmax(pred.predictions, axis=1)

        self._true_values = true_values
        self._pred_values = pred_values

        return true_values, pred_values

    def get_evaluation_info(self):
        '''
        Read the trainer_state.json and return the useful information.

        Return:
            train_accu (list):
                All training accuracies.
            train_loss (list):
                All training losses.
            valid_accu (list): 
                All validation accuracies.
            valid_loss (list):
                All validation losses.
            self._true_values (list):
                True label values.
            self._pred_values (list):
                Predicted label values. 
        '''
        with open("B/results/checkpoint-1170/trainer_state.json", "r") as file:
            data = json.load(file)

        train_accu = [entry['train_accuracy'] for entry in data['log_history'] if 'train_accuracy' in entry]
        train_loss = [entry['train_loss'] for entry in data['log_history'] if 'train_loss' in entry]
        valid_accu = [entry['eval_accuracy'] for entry in data['log_history'] if 'eval_accuracy' in entry]
        valid_loss = [entry['eval_loss'] for entry in data['log_history'] if 'eval_loss' in entry]

        return train_accu, train_loss, valid_accu, valid_loss, self._true_values, self._pred_values
