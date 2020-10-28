import os
import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from BertClassifier import BertClassifier, create_data_list
from BertEmbedder import BertEmbedder
from torch.utils.data import TensorDataset
from rnn import LSTMModel
from cnn import CNNModel
from plot import plot_fit

BERT_CLASSIFIER = "bert_classifier"
RNN_CLASSIFIER = "RNN_classifier"
CNN_CLASSIFIER = "CNN_classifier"


def plot_graphs():
    fit_result = None
    checkpoint_dir = os.path.join('.', 'checkpoints')
    checkpoint_file = os.path.join(checkpoint_dir, CNN_CLASSIFIER)
    if os.path.isfile(checkpoint_file):  # Loading the checkpoints if the models already trained with the same hyperparameters
        fit_result = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        plot_fit(fit_result, 'RNN classifier graph', legend='total')


def run_bert_classifier(config):
    headlines_list, labels = create_data_list(config.input_path)
    bert_classifier = BertClassifier(headlines_list, labels, config, BERT_CLASSIFIER)
    dataset = bert_classifier.get_dataset()
    train_dataloader, test_dataloader = bert_classifier.get_dataloader(dataset)
    fit_result = bert_classifier.fit(train_dataloader, test_dataloader)
    fig, axes = plot_fit(fit_result, 'Bert classifier graph', legend='total')


def run_rnn(config):
    headlines_list, labels = create_data_list(config.input_path)
    bert_embedder = BertEmbedder(headlines_list, labels, config)
    embeddings, labels = bert_embedder.get_word_embeddings()
    dataset = TensorDataset(embeddings, labels)
    model = LSTMModel(config, RNN_CLASSIFIER)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_dataloader, test_dataloader = model.get_dataloader(dataset)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    fit_result = model.fit(train_dataloader, test_dataloader, loss_fn, optimizer)


def run_cnn(config):
    headlines_list, labels = create_data_list(config.input_path)
    bert_embedder = BertEmbedder(headlines_list, labels, config)
    embeddings, labels = bert_embedder.get_sentence_embeddings()
    embeddings = embeddings.unsqueeze(0).permute(1,0,2)
    dataset = TensorDataset(embeddings, labels)
    model = CNNModel(config, checkpoint_file=CNN_CLASSIFIER)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_dataloader, test_dataloader = model.get_dataloader(dataset)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    fit_result = model.fit(train_dataloader, test_dataloader, loss_fn, optimizer)


if __name__ == "__main__":
    config = Config()
    # plot_graphs()
    # run_bert_classifier(config)
    run_cnn(config)

