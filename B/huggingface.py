from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, NamedSplit
import evaluate
import numpy as np

class Transformer():
    def __init__(self, train_df, val_df, test_df):
        id2label = {0:'1', 1:'2', 2:'3', 3:'4', 4:'5', 5:'6', 6:'7', 7:'8', 8:'9', 9:'10'}
        label2id = {'1':0, '2':1, '3':2, '4':3, '5':4, '6':5, '7':6, '8':7, '9':8,'10':9}

        model_name = "distilbert/distilbert-base-uncased"
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 10, id2label=id2label, label2id=label2id)
        self._data_collator = DataCollatorWithPadding(tokenizer=self._tokenizer)
        self._accuracy = evaluate.load("accuracy")

        train_df['label'] = [label2id[str(label)] for label in train_df['label']]
        val_df['label'] = [label2id[str(label)] for label in val_df['label']]
        test_df['label'] = [label2id[str(label)] for label in test_df['label']]

        train_dataset, val_dataset, test_dataset = self._custom_dataset(train_df, val_df, test_df)
        self._tokenized_train_dataset = train_dataset.map(self._auto_tokenizer, batched=True)
        self._tokenized_val_dataset = val_dataset.map(self._auto_tokenizer, batched=True)
        self._tokenized_test_dataset = test_dataset.map(self._auto_tokenizer, batched=True)

        #print(train_dataset['label'][:1000])

    def _custom_dataset(self, train_df, val_df, test_df):
        train_dataset = Dataset.from_pandas(train_df[['label', 'question']], split=NamedSplit('train'), preserve_index=True)
        val_dataset = Dataset.from_pandas(val_df[['label', 'question']], split=NamedSplit('val'), preserve_index=True)
        test_dataset = Dataset.from_pandas(test_df[['label', 'question']], split=NamedSplit('test'), preserve_index=True)

        return train_dataset, val_dataset, test_dataset
    
    def _auto_tokenizer(self, dataset):
        return self._tokenizer(dataset['question'])
    
    def _evaluation_metrics(self, pred_true):
        pred_labels, true_labels = pred_true
        pred_labels = np.argmax(pred_labels, axis=1)
        return self._accuracy.compute(predictions=pred_labels, references=true_labels)
    
    def train(self):
        training_args = TrainingArguments(
            output_dir="B/results",
            overwrite_output_dir=True,
            logging_dir="B/logs",
            num_train_epochs=2,
            per_device_train_batch_size=64,
            per_device_eval_batch_size=64,
            learning_rate=2e-5,
            weight_decay=0.01,
            evaluation_strategy="steps",
            logging_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
        )   
        
        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=self._tokenized_train_dataset,
            eval_dataset=self._tokenized_val_dataset,
            tokenizer=self._tokenizer,
            data_collator=self._data_collator,
            compute_metrics=self._evaluation_metrics,
        )

        trainer.train()