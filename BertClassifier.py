import torch
import tqdm
import time
import sys
import json
import torch.optim as optim
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification


def create_data_list(input_path):
    with open(input_path, 'r') as file_json:
        data = json.load(file_json)

    all_words = []
    all_labels = []
    for key in data:
        headline = data[key]['headline'].lower()

        all_words.append(headline)
        all_labels.append(data[key]['is_sarcastic'])

    return all_words, all_labels


class BertClassifier:
    def __init__(self, headlines_list, labels, config):
        self.config = config
        self.headlines_list = headlines_list
        self.labels = torch.tensor(labels).unsqueeze(1)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True)
        self.batch_size = self.config.batch_size

    def get_dataset(self):
        max_len = 0
        input_ids = []
        attention_masks = []
        for sent in self.headlines_list:
            # from here: this code is just to see what max_length should be. not necessary once we know
            input_ids_len = self.tokenizer.encode(sent, add_special_tokens=True)
            max_len = max(max_len, len(input_ids_len))
            if max_len == len(input_ids_len):
                max_input = sent
            # to here
            encoded_dict = self.tokenizer.encode_plus(
                sent, add_special_tokens=True, max_length=37, padding='max_length',
                return_attention_mask=True, return_tensors='pt',
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(self.labels)

        # Combine the training inputs into a TensorDataset.
        dataset = TensorDataset(input_ids, attention_masks, labels)

        return dataset

    def get_dataloader(self, dataset):

        # Calculate the number of samples to include in each set.
        train_size = int((1-self.config.test_size) * len(dataset))
        test_size = len(dataset) - train_size

        # Divide the dataset by randomly selecting samples.
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=self.batch_size)
        test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=self.batch_size)
        return train_dataloader, test_dataloader

    def fit(self, train_dataloader: DataLoader, test_dataloader: DataLoader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        # model = self.model
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        epochs = self.config.num_epochs
        for epoch_i in range(0, epochs):
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            self.train_epoch(train_dataloader, optimizer, device)
            self.test_epoch(test_dataloader, device)

    def train_epoch(self, train_dataloader, optimizer, device):
        total_train_accuracy = 0
        total_train_loss = 0
        self.model.train()
        pbar_file = sys.stdout
        pbar_name = "train_batch"
        num_batches = len(train_dataloader.batch_sampler)
        with tqdm.tqdm(desc=pbar_name, total=num_batches,
                       file=pbar_file) as pbar:
            for step, batch in enumerate(train_dataloader):
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                self.model.zero_grad()

                # Forward pass
                output = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                # Log the train loss
                loss = output.loss
                logits = output.logits
                total_train_loss += loss.item()

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Weight updates
                optimizer.step()

                logits = logits.detach().cpu()
                y = b_labels.to('cpu')
                y_pred = torch.argmax(logits, dim=1).unsqueeze(1)
                total_train_accuracy += torch.sum(y_pred == y).float().item()
                pbar.set_description(f'{pbar_name} ({loss.item():.3f})')
                pbar.update()
        avg_train_accuracy = total_train_accuracy / len(train_dataloader)
        print("  Training accuracy: {0:.2f}".format(avg_train_accuracy))
        avg_train_loss = total_train_loss / len(train_dataloader)
        # Log the Avg. train loss
        print("  Training loss: {0:.2f}".format(avg_train_loss))

    def test_epoch(self, test_dataloader, device):
        self.model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0
        # Evaluate data for one epoch
        pbar_file = sys.stdout
        pbar_name = "test_batch"
        num_batches = len(test_dataloader.batch_sampler)
        with tqdm.tqdm(desc=pbar_name, total=num_batches,
                       file=pbar_file) as pbar:
            for batch in test_dataloader:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                with torch.no_grad():
                    output = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                    loss = output.loss
                    logits = output.logits
                    total_eval_loss += loss.item()
                    logits = logits.detach().cpu()
                    y = b_labels.to('cpu')
                    y_pred = torch.argmax(logits, dim=1).unsqueeze(1)
                total_eval_accuracy += torch.sum(y_pred == y).float().item()
                pbar.set_description(f'{pbar_name} ({loss.item():.3f})')
                pbar.update()

        avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
        print("  Validation accuracy: {0:.2f}".format(avg_val_accuracy))
        avg_val_loss = total_eval_loss / len(test_dataloader)
        # Log the Avg. validation accuracy
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))

    def get_bert_classification(self):
        for i in range(0, len(self.head_lines_list), self.batch_size):
            batch = self.head_lines_list[i:i+self.batch_size]
            encoded = self.tokenizer(batch, return_tensors='pt', padding=True)
            encoded = torch.tensor(encoded)
            target = torch.tensor(self.labels[i:i+self.batch_size])
            # inp = batch.cuda()
            # target = target.cuda()
            output = self.model(batch, target)
            loss = output.loss
            print(f"loss: {loss}")
            logits = output.logits
            print(f"logits:{logits}")
