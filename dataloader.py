import numpy as np
from tqdm import tqdm
from transformers.modeling_outputs import Seq2SeqModelOutput
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, Trainer, TrainingArguments, BertGenerationDecoder
import torch
import glob
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

class CustomDataset(Dataset):

    def __init__(self, path_data):
        self.path_data = path_data
        path_X = path_data+ "X.pt"
        path_X_mask = path_data+ "X_mask.pt"
        path_y_ids = path_data+ "y_ids.pt"
        path_lm_label = path_data+ "lm_labels.pt"
        self.X = torch.load(path_X)
        self.X_mask = torch.load(path_X_mask)
        self.y_ids = torch.load(path_y_ids)
        self.lm_labels = torch.load(path_lm_label)
        # print(self.label.shape)
        # self.label_embedded = torch.load("label_embedded.pt")

    

    def clean_data(self,sentences):
      sentences = sentences.lower()
      clean = re.sub(r'[!"#$%&()*+«»,-./:;<=>?@[\]^`{|}~]\s*', " ", sentences)
      clean = clean.strip()
      clean = " ".join(clean.split())
      return clean

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # print(self.data.shape)
        input_ids = torch.stack(self.X[index])
        input_ids = input_ids.to(torch.long)
        input_mask = torch.Tensor(self.X_mask[index])
        return input_ids.squeeze(1),input_mask,self.y_ids[index].squeeze(0).long(), self.lm_labels[index].squeeze(0).long()

