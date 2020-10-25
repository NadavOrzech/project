from config import Config
from BertClassifier import BertClassifier, create_data_list
from BertEmbedder import BertEmbedder
import torch


if __name__ == "__main__":
    config = Config()
    # dataloader = Dataloader(config)
    # tmp = create_data_list(config.input_path)
    # for l in tmp[0]:
    #     if len(l) == 38:
    #         print(l)
    # max_sen = max(len(l) for l in tmp[0])
    headlines_list, labels = create_data_list(config.input_path)
    bert_embedder = BertEmbedder(headlines_list, labels, config)
    embeddings = bert_embedder.get_word_embeddings()
    # bert_classifier = BertClassifier(headlines_list, labels, config)
    # dataset = bert_classifier.get_dataset()
    # train_dataloader, test_dataloader = bert_classifier.get_dataloader(dataset)
    # bert_classifier.fit(train_dataloader, test_dataloader)
