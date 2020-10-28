from config import Config
from BertClassifier import BertClassifier, create_data_list
from BertEmbedder import BertEmbedder
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.optim as optim
from rnn import LSTMModel
import torch
from plot import plot_fit
import os

if __name__ == "__main__":
    config = Config()
    fit_result = None
    checkpoint_dir = os.path.join('.', 'checkpoints')
    # checkpoint_file = os.path.join(checkpoint_dir, 'bert classifier')
    checkpoint_file = os.path.join(checkpoint_dir, 'RNN classifier')

    if os.path.isfile(checkpoint_file):  # Loading the checkpoints if the models already trained with the same hyperparameters
        fit_result = torch.load(checkpoint_file, map_location=torch.device('cpu'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        plot_fit(fit_result, 'RNN classifier graph', legend='total')
    dataloader = Dataloader(config)
    tmp = create_data_list(config.input_path)
    for l in tmp[0]:
        if len(l) == 38:
            print(l)
    max_sen = max(len(l) for l in tmp[0])
   # else:
        # headlines_list, labels = create_data_list(config.input_path)
        # bert_embedder = BertEmbedder(headlines_list, labels, config)
        # embeddings = bert_embedder.get_word_embeddings()
        # bert_classifier = BertClassifier(headlines_list, labels, config, 'bert classifier')
        # dataset = bert_classifier.get_dataset()
        # train_dataloader, test_dataloader = bert_classifier.get_dataloader(dataset)
        # fit_result = bert_classifier.fit(train_dataloader, test_dataloader)
        # fig, axes = plot_fit(fit_result, 'Attention_graph', legend='total')

    headlines_list, labels = create_data_list(config.input_path)
    bert_embedder = BertEmbedder(headlines_list, labels, config)
    embeddings, labels = bert_embedder.get_word_embeddings()
    dataset = TensorDataset(embeddings, labels)
    model = LSTMModel(config)
    model.to(device)

    train_dataloader, test_dataloader = model.get_dataloader(dataset)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    #
    fit_result = model.fit(train_dataloader, test_dataloader, loss_fn, optimizer)
