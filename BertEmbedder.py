import torch
import numpy as np
from transformers import BertTokenizer, BertModel

MAX_LEN = 66


class BertEmbedder:
    def __init__(self, headlines_list, labels, config):
        self.config = config
        self.headlines_list = headlines_list
        self.labels = torch.tensor(labels)  # .unsqueeze(1)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

    def get_word_embeddings(self):
        """
        Creates word embeddings for all the samples
        Currently does this by summing the vectors from the last 4 hidden layers
        Other optional approaches: concating the last 4 hidden layers (drawback - creates a larger embedding),
        taking the second to last layer and more..
        :return: sent_vecs_sum - a t
        """
        input_ids = []
        max_len = MAX_LEN
        for sent in self.headlines_list:
            encoded_dict = self.tokenizer.encode_plus(
                sent, add_special_tokens=True, max_length=max_len, padding='max_length',
                return_attention_mask=True, return_tensors='pt',
            )
            input_ids.append(encoded_dict['input_ids'])
        input_ids = torch.cat(input_ids, dim=0)

        # Put the model in "evaluation" mode, meaning feed-forward operation.
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids)
            hidden_states = outputs[2]
        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = token_embeddings.permute(1, 2, 0, 3)
        sent_vecs_sum = np.zeros((token_embeddings.shape[0], max_len, 768)) # in the end should be size (num_samples, max_length, 768)
        for i, sentence in enumerate(token_embeddings):
            for j, token in enumerate(sentence):
                # `token` is a [13 x 768] tensor
                # Sum the vectors from the last four layers
                sum_vec = torch.sum(token[-4:], dim=0)
                sent_vecs_sum[i, j, :] = sum_vec

        sent_vecs_sum = torch.tensor(sent_vecs_sum, dtype=torch.float32)
        return sent_vecs_sum, self.labels

    def get_sentence_embeddings(self):
        input_ids = []
        max_len = MAX_LEN
        for sent in self.headlines_list:
            encoded_dict = self.tokenizer.encode_plus(
                sent, add_special_tokens=True, max_length=max_len, padding='max_length',
                return_attention_mask=True, return_tensors='pt',
            )
            input_ids.append(encoded_dict['input_ids'])
        input_ids = torch.cat(input_ids, dim=0)

        # Put the model in "evaluation" mode, meaning feed-forward operation.
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids)
            hidden_states = outputs[2]
        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = token_embeddings.permute(1, 0, 2, 3)
        sent_vecs_avg = np.zeros((token_embeddings.shape[0], 768))  # in the end should be size (num_samples, 768)
        for i, sentence in enumerate(token_embeddings):
            token_vecs = sentence[-2]
            # Calculate the average of all token vectors in the sentence.
            sentence_embedding = torch.mean(token_vecs, dim=0)
            sent_vecs_avg[i, :] = sentence_embedding

        sent_vecs_avg = torch.tensor(sent_vecs_avg, dtype=torch.float32)
        return sent_vecs_avg, self.labels


