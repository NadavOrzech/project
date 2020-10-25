from enum import Enum

class Config():
    def __init__(self):
        # self.input_path = "C:\\Users\\Dell\\Desktop\\Technion\\Project\\news-headlines-dataset-for-sarcasm-detection\\sarcasm_dataset.json"
        self.input_path = ".\\sarcasm_dataset_very_small.json"
        self.num_features = 10
        self.num_epochs = 5
        self.batch_size = 128
        self.lr = 2e-5
        self.test_size = 0.1
        self.features_map = {
            'punctuations': 0,
            'quotes': 1,
            'polarity': 2,
            'subjectivity': 3,
            'intersifier': 4,
            'interjection': 5,
            'sentence_length': 6,

        }
        self.interjections = ["cheers",
                            "aha",
                            "wow",
                            "holy",
                            "bam",
                            "bravo",
                            "rats",
                            "yikes",
                            "ahem",
                            "oops",
                            "whoa",
                            "well",
                            "er",
                            "god",
                            "wow",
                            "hum",
                            "oh",
                            "ugh",
                            "yeah",
                            "pew",
                            "eh",
                            "hi",
                            "hello",
                            "great",
                            "er",
                            "goodness",
                            "shoot",
                            "hush",
                            "nice",
                            "huh",
                            "sweet",
                            "ugh",
                            "yo",
                            "cool",
                            "yum"]
        self.intensifiers = ["very",
                            "so", 
                            "really",
                            "extremely",
                            "enormous",
                            "huge",
                            "tiny",
                            "brilliant",
                            "awful",
                            "terrible",
                            "disgusting",
                            "dreadful",
                            "certain",
                            "excellent",
                            "perfect",
                            "perfectly",
                            "wonderful",
                            "splendid",
                            "delicious",
                            "absolutely",
                            "completely",
                            "exceptionally",
                            "particularly",
                            "really",
                            "totally",
                            "seriously", 
                            "dangerously", 
                            "bitterly",
                            "much",
                            "lot",
                            "far",
                            "complete",
                            "ever",
                            "deadly",
                            "highly",
                            "furiously",
                            "deeply",
                            "fully",
                            "quite",
                            "pretty",
                            "horribly",
                            "madly",
                            "sick",
                            "bloody",
                            "fucking",
                            "crazy",
                            "deadly",
                            "terribly",
                            "super",
                            "terrifically",
                            "fantastically",
                            "insane",
                            "insanely",
                            "strikingly",
                            "extra",
                            "amazing",
                            "amazingly",
                            "ridiculous",
                            "absurd",
                            "ridiculously",
                            "unusually",
                            "incredibly",
                            "especially",
                            "terrified",
                            "furious",
                            "freezing",
                            "superb",
                            "beautiful",
                            "idiot",
                            "exhausted",
                            "anxious",
                            "literally",
                            "definitely",
                            "certainly",
                            "unbelievably",
                            "fascinating",
                            "exhausting",
                            "hilarious",
                            "gorgeous",
                            "clearly",
                            "special",
                            "astoundingly",
                            "phenomenally",
                            "strongly",
                            "fairly",
                            "wonderfully",
                            "greatly",
                            "hugely",
                            "suspiciously",
                            "hardly",
                            "mostly"]


# class FeaturesMap(Enum):
#     PUNCTUATIONS = 0
