import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
# from transformers.models.gpt2.modeling_gpt2 import GPT2Model
# from transformers import GPT2Tokenizer
from transformers import BertTokenizer, BertModel
from einops import rearrange
from embed import DataEmbedding, DataEmbedding_wo_time
# from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from layers.MultiscaleCNN import MultiScaleCNN
from layers.heatmap import TimeSeriesToHeatmap
from layers.Cross_Attention import CrossAttention

class LLM_MRL(nn.Module):

    def __init__(self, configs, device):
        super(LLM_MRL, self).__init__()
        self.config = configs   
        self.pretrain = configs.pretrain
        self.convert_to_image = TimeSeriesToHeatmap(clip_percentage=0.005)
        self.MultiConv = MultiScaleCNN(in_channels=3, out_channels=configs.d_model)
        self.time_embedding = nn.Linear(configs.seq_len, configs.d_model)
        time_encoder_layer = nn.TransformerEncoderLayer(
            d_model=configs.d_model,
            nhead=configs.n_heads,
            dim_feedforward=configs.d_ff,
            dropout=configs.dropout,
            batch_first=True
        )
        self.time_encoder = nn.TransformerEncoder(time_encoder_layer, num_layers=2)
        self.max_input_text_length = configs.prompt_max_len
        self.proj = nn.Linear(configs.seq_len*configs.vars, configs.vars)
        
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_attentions=True, output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        for param in self.bert.parameters():
            param.requires_grad = False


        self.CrossAttention = CrossAttention(configs.d_model, num_heads=8)
        self.out_layer = nn.Linear(configs.d_model, configs.pred_len)
        
        for layer in (self.time_embedding, self.time_encoder, self.out_layer):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0


    def forward(self, x, itr):
    
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        x_time = self.time_embedding(x)  # [B, vars, d_model]
        x_time = self.time_encoder(x_time)  # [B, vars, d_model]
       

        x_image = rearrange(x, 'b m l -> b l m')
        x_image = self.convert_to_image(x_image)
        x_image = x_image.repeat(1, 3, 1, 1)
        x_image = self.MultiConv(x_image)  # [B, d_model, H, W]
        x_image = x_image.flatten(2)  # [B, d_model, H*W]

        x_image = self.proj(x_image)  # [B, d_model, seq_len]
        x_image = rearrange(x_image, 'b d m -> b m d')  # [B, seq_len, d_model]
        

        x_text = self.prompt_construction(x, self.config.description, self.config.pred_len, self.config.seq_len)  # List of prompts for each batch
        x_text = self.tokenizer(
            x_text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(x.device)
        with torch.no_grad():
            x_text = self.bert(**x_text).last_hidden_state  # [B, hidden_size]

        query = x_time + x_image  # [B, vars, d_model]
       
        outputs = self.CrossAttention(query, x_text, x_text)+query
       
        
        outputs = self.out_layer(outputs)
        outputs = rearrange(outputs, 'b m l -> b l m', b=B)

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs
    
    def prompt_construction(self, x_enc, description, pred_len, seq_len, top_k=5):
        """
        Generate text prompts for the language model based on time series data.
        Each variable in the time series will have its own prompt.
        """
        B, n_vars, T = x_enc.shape  # [batch_size, variables, sequence_length]

        # Initialize a list to store prompts for each batch
        prompts = []
    
        # Calculate overall statistics for each batch
        for b in range(B):
            # Calculate statistics for the current batch
            min_value = torch.min(x_enc[b]).item()  # Overall minimum value for the batch
            max_value = torch.max(x_enc[b]).item()  # Overall maximum value for the batch
            median_value = torch.median(x_enc[b]).item()  # Overall median value for the batch
            trend = x_enc[b].diff(dim=-1).sum().item()  # Difference along the time dimension

            # Determine the overall trend direction
            trend_direction = "upward" if trend > 0 else "downward"
                
            prompt_parts = [
                "The time series is encoded by a Transformer encoder with two layers, while its heatmap is processed by a multi-scale CNN for forecasting.",
                f"Dataset: {description}",
                f"Task: Forecast the next {pred_len} steps using the past {seq_len} steps.",
                f"Input statistics: min value = {min_value:.3f}, max value = {max_value:.3f}, median value = {median_value:.3f}, the overall trend is {trend_direction}."
            ]
            prompt = " ".join(prompt_parts)
            prompt = prompt[:self.max_input_text_length] if len(prompt) > self.max_input_text_length else prompt
            prompts.append(prompt)  

        return prompts
