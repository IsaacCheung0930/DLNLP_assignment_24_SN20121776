import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import random

class CustomDataloader():
    def __init__(self, data_list):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = get_tokenizer("basic_english")
        self._train_list, self._val_list, self._test_list = self._split_data(data_list)
        self._vocab_builder(self._train_list)

    def _split_data(self, data_list):
        data_size = len(data_list)
        train_size = int(0.8 * data_size)
        test_size  = (data_size -train_size)//2
        val_size = data_size - train_size - test_size
        train_list, val_list, test_list = random_split(data_list, [train_size, test_size, val_size])

        return train_list, val_list, test_list
    
    def _word_to_tokens(self, data_list):
        for _, _, question, _ in data_list:
            yield self._tokenizer(question)
    
    def _vocab_builder(self, data_list):
        self.vocab = build_vocab_from_iterator(self._word_to_tokens(data_list), specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])

    def _collate_batch(self, batch):
        label_list, text_list, offsets = [], [], [0]

        for _, label, question, _ in batch:
            label_list.append(self.pipelines("label", label))
            indexed_text = torch.tensor(self.pipelines("text", question), dtype=torch.int64)

            text_list.append(indexed_text)
            offsets.append(indexed_text.size(0))

        label_list = torch.tensor(label_list, dtype=torch.int64)
        text_list = torch.cat(text_list)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)

        return label_list.to(self._device), text_list.to(self._device), offsets.to(self._device)
    
    def _collate_batch_padding(self, batch):
        label_list, text_list, seq_lengths = [], [], []
        max_seq_length = max(len(text) for _, _, text, _ in batch)

        for _, label, question, _ in batch:
            label_list.append(self.pipelines("label", label))
            indexed_text = torch.tensor(self.pipelines("text", question), dtype=torch.int64)

            padded_text = torch.nn.functional.pad(indexed_text, (0, max_seq_length - len(indexed_text))).tolist()
            text_list.append(padded_text)
            seq_lengths.append(len(indexed_text))

        label_list = torch.tensor(label_list, dtype=torch.int64)
        text_list = torch.tensor(text_list)
        seq_lengths = torch.tensor(seq_lengths, dtype=torch.int64)

        return label_list.to(self._device), text_list.to(self._device), seq_lengths.to(self._device)
    
    def pipelines(self, type, input):
        if type == "text":
            return self.vocab(self._tokenizer(input))
        else:
            return int(input) - 1
        
    def get_dataloader(self, padding):
        batch_size = 64
        if padding:
            collate_fn = self._collate_batch_padding
        else:
            collate_fn = self._collate_batch

        train_dataloader = DataLoader(to_map_style_dataset(self._train_list), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        valid_dataloader = DataLoader(to_map_style_dataset(self._val_list), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        test_dataloader = DataLoader(to_map_style_dataset(self._test_list), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        return train_dataloader, valid_dataloader, test_dataloader
    
    def get_pipelines(self, type, input):
        return self._pipelines(type, input)
    
    def get_test_samples(self, num_samples):
        sample_indices = random.sample(range(len(self._test_list)), num_samples)
        test_samples = Subset(self._test_list, sample_indices)
        sample_dataloader = DataLoader(to_map_style_dataset(test_samples), batch_size=num_samples, shuffle=False, collate_fn=self._collate_batch_padding)

        return list(test_samples), sample_dataloader
