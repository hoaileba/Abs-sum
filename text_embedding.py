from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Optional
from attention import Attention
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from dataloader import CustomDataset
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (BaseModelOutput,
                                            BaseModelOutputWithPastAndCrossAttentions,
                                            CausalLMOutputWithCrossAttentions,
                                            Seq2SeqLMOutput,
                                            Seq2SeqModelOutput,)
from transformers.modeling_outputs import Seq2SeqModelOutput,Seq2SeqLMOutput
from transformers import T5ForConditionalGeneration, PegasusTokenizer, Trainer, TrainingArguments, BertGenerationDecoder

model_name = 'NlpHUST/t5-small-vi-summarization'

model = T5ForConditionalGeneration.from_pretrained(model_name)
# model_decoder = model.model.decoder.to("cuda")
# model_encoder = model.model.encoder
config = model.config
class SentEmbedding(nn.Module):
  def __init__(self, encoder,decoder,query_dim,word_embedding_dim):
    super(SentEmbedding, self).__init__()
    self.decoder = decoder
    self.attention_sent = Attention(word_embedding_dim,query_dim).to("cuda")
    self.dropout = nn.Dropout(0.25)
    # self.CNN = nn.Conv2d(1,
    #                          512, (3, 512),
    #                          padding=(int((3 - 1) / 2), 0))
  def forward(self,inputs):
    """
      inputs: đầy vào là các vector của từ đã được encode
        shape: (batch_size, n, 512)
      outputs:
        vector feature duy nhất của 1 câu 
        shape: (batch_size, 1, 512)
    """
    # print("inputs_text: ",inputs.shape)
    # convoluted_text_vector = self.CNN(
    #         inputs.unsqueeze(dim=1)).squeeze(dim=3)
    # activated_text_vector = F.relu(convoluted_text_vector)
    # activated_text_vector = self.dropout(activated_text_vector)
    attention_vec = self.attention_sent(inputs)
 
    return attention_vec



class SectionEmbedding(nn.Module):
  def __init__(self, encoder,decoder,query_dim, max_sent,sequen_embedding_dim):
    super(SectionEmbedding, self).__init__()
    self.encoder = encoder
    self.decoder = decoder

    # print("max_length: ",max_sent)
    # 1 đoạn nhiều cầu cần nhiều khối tính feature của câu 
    self.sent_encoders = nn.ModuleDict({
            "encoder_sentence_"+str(i):
            SentEmbedding(None,None,query_dim,sequen_embedding_dim).to("cuda")
            for i in range(max_sent)
        })
    self.attention_sec = Attention(sequen_embedding_dim,query_dim).to("cuda")
    # print(query_dim)

  def forward(self,
            inputs_section = None, 
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            inputs_embeds=None,
            head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None):
    """
        inputs: 
          đầu vào là cả đoạn
          shape: (batch_size,n,m)
          (n,bs,m,512)
          n,batch,m
                n: số câu 
                m: số từ trong câu
        outputs:
          vector feature duy nhất của 1 câu 
          shape: (batch_size, 1, 512)
    """
    
      
    text_embed = [ self.encoder(input_id.to("cuda"))[0] for input_id in inputs_section.transpose(0,1) ]
    text_embed = torch.stack(text_embed)
    # print("Text embedd: ",text_embed.shape)
    # label = inputs_section[1]

    text_vectors = [
        self.sent_encoders[name](text_embed[id].to("cuda"))
        for id, name in enumerate(self.sent_encoders)
    ]
    attention_sec_vec = torch.stack(text_vectors, dim=1)
    # print("attention_sec_vec: ",attention_sec_vec.shape)
    # attention_sec_vec = self.attention_sec(attention_sec_vec.to("cuda"))
    # attention_sec_vec = attention_sec_vec.unsqueeze(1)
    del text_embed
    del text_vectors
    return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=attention_sec_vec,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
    )



if __name__ == "__main__":
  a = torch.zeros((2,50,100))
  sent = SectionEmbedding(a)
  print(sent[0].shape)
