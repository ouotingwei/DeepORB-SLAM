#!/usr/bin/env python3
from transformers import BertForTokenClassification, BertModel, BertTokenizer, BertConfig,get_linear_schedule_with_warmup
from transformers.models.bert.modeling_bert import BertEmbeddings
import torch.nn as nn
import torch
import pickle
from datasets import Dataset, load_metric
from transformers import BertTokenizerFast, BertForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import warnings
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import matplotlib.pyplot as plt
import time
warnings.filterwarnings("ignore")


class CustomBertEmbeddings(BertEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        # Remove word_embeddings, position_embeddings, and token_type_embeddings
        # self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.word_embeddings=None
        self.position_embeddings=None
        self.token_type_embeddings=None
        # self.fc=nn.Linear(5, config.hidden_size) 
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        # inputs_embeds should be provided directly
        if inputs_embeds is None:
            raise ValueError("inputs_embeds must be provided as word_embeddings is removed.")
        
        # embeddings = self.fc(inputs_embeds)
        embeddings = inputs_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class CustomBertForTokenClassification(BertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.bert.embeddings = CustomBertEmbeddings(config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        if inputs_embeds is None:
            raise ValueError("inputs_embeds must be provided as word_embeddings is removed.")
        return super().forward(input_ids=None, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, labels=labels)

config = BertConfig.from_pretrained('bert-base-uncased')
config.type_vocab_size = 2 # Adjust this based on your classification task
config.num_hidden_layers = 12
config.hidden_size=4
config.num_attention_heads=4
config.intermediate_size=config.hidden_size*1000
config.max_position_embeddings=1024
config.initializer_range=0.1
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
# Initialize the custom model
model = CustomBertForTokenClassification(config)

import socket
import struct
import numpy as np
from torch.utils.data import DataLoader

def receive_data_from_FSM(server_socket):
    # 接收行數和列數
    rows_bytes = server_socket.recv(4)
    cols_bytes = server_socket.recv(4)
    
    rows = int.from_bytes(rows_bytes, byteorder='little')
    cols = int.from_bytes(cols_bytes, byteorder='little')

    # 接收數據
    data = []
    for i in range(rows):
        row_data_bytes = server_socket.recv(cols * 4)  # 假設數據類型為浮點數（4個字節）
        row_data = np.frombuffer(row_data_bytes, dtype=np.float32)
        data.append(row_data)
    
    # 將列表轉換為NumPy數組
    data_array = np.array(data)

    return data_array

def send_data_to_ORB(client_socket, model_output, keep_ratio):
    threshold = np.percentile(model_output, 100 - (keep_ratio*100))
    
    label_list = []
    for i in range(len(model_output)):
        if model_output[i] >= threshold:
            label_list.append(1)
        else:
            label_list.append(0)

    #print(label_list)
    
    label_array = label_list
    
    rows = len(label_array)
    cols = 1 if rows > 0 else 0

    # 發送行數和列數
    client_socket.send(struct.pack('i', rows))
    client_socket.send(struct.pack('i', cols))

    # 發送數據
    for value in label_array:
        # 將單個浮點數打包成二進制數據並發送
        client_socket.send(struct.pack('f', value))



def main():
    global model
    model = torch.load("/home/wei/orbslam_selector/src/DeepModule/selector_weight/best_epoch1232")
    model.eval()

    # GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('127.0.0.1', 8888))
    server_socket.listen(1)

    while True:
        client_socket, _ = server_socket.accept()
        received_data = receive_data_from_FSM(client_socket)
        
        input_tensor = torch.tensor(received_data).unsqueeze(0)
        #print(input_tensor.shape)

        numKp = len(received_data)

        # data -> gpu
        input_tensor = input_tensor.to(device)

        output_bel = []
        # Step 3: Perform inference
        with torch.no_grad():
            scores = model(inputs_embeds=input_tensor)
            #print(scores.logits)
            for i in range(numKp):
                bel_ = np.array([float(scores.logits[0, i, :][0]), float(scores.logits[0, i, :][1])])
                probabilities = 1 / (1 + np.exp(bel_))
                good_bel = probabilities[1]
                output_bel.append(good_bel)

            send_data_to_ORB(client_socket, output_bel, keep_ratio=0.4)

if __name__ == "__main__":
    main()
