import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

# Selecting the Device
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_device()


def nirmalya_softmax(input_tensor):
    eps = 1e-10
    p1 = torch.exp(input_tensor)
    p2 = p1.sum(dim=-1, keepdim=True) + eps
    output_tensor = p1/p2
    return output_tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
    
    def forward(self):
        even_i = torch.arange(0, self.d_model, 2, dtype=torch.float32).to(device)
        denominator = torch.pow(10000,even_i/self.d_model).to(device)
        position = torch.arange(0, self.max_sequence_length, 1, dtype=torch.float32).to(device)
        position = position.reshape(self.max_sequence_length, 1).to(device)
        even_PE = torch.sin(position/denominator).to(device)
        odd_PE = torch.cos(position/denominator).to(device)
        stacked_PE = torch.stack([even_PE, odd_PE], dim=2).to(device)
        PE = torch.flatten(stacked_PE, start_dim=1, end_dim=2).to(device)
        return PE


class SentenceEmbedding(nn.Module):
    def __init__(self,max_sequence_length,d_model,language_to_index,START_TOKEN,END_TOKEN,PADDING_TOKEN):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.language_to_index = language_to_index
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN
        self.vocabulary_size = len(language_to_index)
        self.embedding = nn.Embedding(self.vocabulary_size, d_model)
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout = nn.Dropout(p = 0.1)
    
    # Function which Performs the Tokenization Operation For A Single Sentence
    def tokenize(self,sentence,start_token,end_token):
        sentence_letter_number = [self.language_to_index[i] for i in list(sentence)]
        if start_token:
            sentence_letter_number.insert(0, self.language_to_index[self.START_TOKEN])
        if end_token:
            sentence_letter_number.append(self.language_to_index[self.END_TOKEN])
        for j in range(len(sentence_letter_number), self.max_sequence_length, 1):
            sentence_letter_number.append(self.language_to_index[self.PADDING_TOKEN])
        return sentence_letter_number
    
    # Function which Performs the Tokenization Operation For A Batch of Sentences
    def batch_tokenize(self,batch,start_token,end_token):
        tokenized = []
        for i in range(0, len(batch), 1):
            conversion = self.tokenize(batch[i],start_token=start_token,end_token=end_token)
            tokenized.append(conversion)
        return torch.tensor(tokenized)
    
    def forward(self,x,start_token,end_token):
        x = self.batch_tokenize(x,start_token,end_token).to(device)
        x = (self.embedding(x))*(math.sqrt(self.d_model))
        position = ((self.position_encoder()).requires_grad_(False)).to(device)
        x = self.dropout(x + position)
        return x


# Calculation of Scaled Dot Product Attention
def Scaled_Dot_Product_Attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    scaled_dot_product = (torch.matmul(q, k.transpose(-2,-1))/math.sqrt(d_k)).to(device)
    if mask is not None:
        scaled_dot_product = scaled_dot_product.permute(1,0,2,3)
        scaled_dot_product = scaled_dot_product + mask.to(device)
        scaled_dot_product = scaled_dot_product.permute(1,0,2,3)
    attention = nirmalya_softmax(scaled_dot_product)
    out = torch.matmul(attention, v).to(device)
    return attention, out


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model//num_heads
        self.q_layer = nn.Linear(d_model, d_model, bias = False)
        self.k_layer = nn.Linear(d_model, d_model, bias = False)
        self.v_layer = nn.Linear(d_model, d_model, bias = False)
        self.linear_layer = nn.Linear(d_model, d_model, bias = False)
    
    def forward(self, x, mask):
        batch_size, sequence_length, d_model = x.shape
        #print(f'Size of input tensor: {batch_size, sequence_length, d_model}')

        # Generation of q Tensor
        q = self.q_layer(x)
        #print(q.shape)
        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        #print(q.shape)
        q = q.permute(0, 2, 1, 3)
        # Now the Shape will be (batch_size, num_heads, sequence_length, head_dim)
        #print(q.shape)
        #print('**')

        # Generation of k Tensor
        k = self.k_layer(x)
        #print(k.shape)
        k = k.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        #print(k.shape)
        k = k.permute(0, 2, 1, 3)
        # Now the Shape will be (batch_size, num_heads, sequence_length, head_dim)
        #print(k.shape)
        #print('***')

        # Generation of v Tensor
        v = self.v_layer(x)
        #print(v.shape)
        v = v.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        #print(v.shape)
        v = v.permute(0, 2, 1, 3)
        # Now the Shape will be (batch_size, num_heads, sequence_length, head_dim)
        #print(v.shape)
        #print('V\n',v)
        #print('****')

        # Calculation of Multi Head Attention
        (attention, attention_head)= Scaled_Dot_Product_Attention(q, k, v, mask = mask)
        # Shape of Attention Probability will be (batch_size, num_heads, sequence_length, sequence_length)
        #print(attention.shape)
        # Shape of Attention Head will be (batch_size, num_heads, sequence_length, head_dim)
        #print(attention_head.shape)
        #print('Attention\n',attention)
        #print('Attention Head\n',attention_head)

        # Concatination of Multiple Heads
        attention_head = attention_head.permute(0, 2, 1, 3)
        # Now the Shape of Attention Head will be (batch_size, sequence_length, num_heads, head_dim)
        #print(attention_head.shape)
        #print(attention_head)
        attention_head = attention_head.reshape(batch_size, sequence_length, self.num_heads*self.head_dim)
        # Now the Shape of Attention Head will be (batch_size, sequence_length, d_model)
        #print(attention_head.shape)
        #print(attention_head)

        # Inter Communication between Multiple Heads
        z = self.linear_layer(attention_head)
        # Shape of z tensor will be (batch_size, sequence_length, d_model)
        #print(z.shape)
        return z


class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape):
        super(LayerNormalization, self).__init__()
        self.parameters_shape = parameters_shape
        self.eps = 0.00001
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))
    
    def forward(self, x):
        if len(self.parameters_shape)==1:
            dims = [-1]
        else:
            dims = [-2, -1]
        mean_values = (x.mean(dim=dims, keepdim=True)).to(device)
        variance_values = (x.var(dim=dims, unbiased=False, keepdim=True)).to(device)
        out = (((x-mean_values)/torch.sqrt(variance_values + self.eps))*self.gamma + self.beta).to(device)
        return out


class PositionwiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model, hidden, drop_prob = 0.1):
        super(PositionwiseFeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(in_features = d_model, out_features = hidden)
        self.linear2 = nn.Linear(in_features = hidden, out_features = d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = drop_prob)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x



class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, hidden, drop_prob = 0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiheadAttention(d_model = d_model, num_heads = num_heads)
        self.dropout1 = nn.Dropout(p = drop_prob)
        self.norm1 = LayerNormalization(parameters_shape = [d_model])
        self.ffn = PositionwiseFeedForwardNetwork(d_model=d_model,hidden=hidden,drop_prob=drop_prob)
        self.dropout2 = nn.Dropout(p = drop_prob)
        self.norm2 = LayerNormalization(parameters_shape = [d_model])
    
    def forward(self, x, mask):
        residual_x = x.clone()
        x = self.attention(x, mask)
        x = self.dropout1(x)
        x = self.norm1(x + residual_x)
        residual_x = x.clone()
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)
        return x


class SequentialEncoderLayersGeneration(nn.Sequential):
    def forward(self, x, mask):
        for i in self._modules.values():
            x = i(x, mask)
        return x



class Encoder(nn.Module):
    def __init__(self,
                d_model,
                num_heads,
                hidden,
                num_layers,
                max_sequence_length,
                language_to_index,
                START_TOKEN,
                END_TOKEN,
                PADDING_TOKEN,
                drop_prob = 0.1):
        super(Encoder, self).__init__()
        self.input_sentence_embedding = SentenceEmbedding(max_sequence_length,d_model,language_to_index,START_TOKEN,END_TOKEN,PADDING_TOKEN)
        self.layers = SequentialEncoderLayersGeneration(*[EncoderLayer(d_model,num_heads,hidden,drop_prob=drop_prob) 
                                      for i in range(num_layers)])
    
    def forward(self, x, mask, encoder_start_token, encoder_end_token):
        x = self.input_sentence_embedding(x, encoder_start_token, encoder_end_token)
        x = self.layers(x, mask)
        return x



# Implementation of Multi Head Cross Attention (Encoder-Decoder Attention)
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model//num_heads
        self.q_layer_ca = nn.Linear(d_model, d_model, bias = False)
        self.k_layer_ca = nn.Linear(d_model, d_model, bias = False)
        self.v_layer_ca = nn.Linear(d_model, d_model, bias = False)
        self.linear_layer_ca = nn.Linear(d_model, d_model, bias = False)
    
    def forward(self, x, y, mask):
        # Tensor x should take value from output of last encoder layer
        # Tensor y should take value from previous sublayer of decoder layer
        batch_size, sequence_length, d_model = x.shape

        # Generation of k Tensor
        k = self.k_layer_ca(x)
        #print(k.shape)
        k = k.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        #print(k.shape)
        k = k.permute(0, 2, 1, 3)
        # Now the Shape will be (batch_size, num_heads, sequence_length, head_dim)
        #print(k.shape)
        #print('***')

        # Generation of v Tensor
        v = self.v_layer_ca(x)
        #print(v.shape)
        v = v.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        #print(v.shape)
        v = v.permute(0, 2, 1, 3)
        # Now the Shape will be (batch_size, num_heads, sequence_length, head_dim)
        #print(v.shape)
        #print('V\n',v)
        #print('****')

        # Generation of q Tensor
        q = self.q_layer_ca(y)
        #print(q.shape)
        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        #print(q.shape)
        q = q.permute(0, 2, 1, 3)
        # Now the Shape will be (batch_size, num_heads, sequence_length, head_dim)
        #print(q.shape)
        #print('**')

        # Calculation of Multi Head Cross Attention
        (cross_attention, cross_attention_head)= Scaled_Dot_Product_Attention(q, k, v, mask = mask)
        # Shape of Attention Probability will be (batch_size, num_heads, sequence_length, sequence_length)
        #print(cross_attention.shape)
        # Shape of Attention Head will be (batch_size, num_heads, sequence_length, head_dim)
        #print(cross_attention_head.shape)
        #print('Cross Attention\n',cross_attention)
        #print('Cross Attention Head\n',cross_attention_head)

        # Concatination of Multiple Heads of Cross Attention
        cross_attention_head = cross_attention_head.permute(0, 2, 1, 3)
        # Now the Shape of Cross Attention Head will be (batch_size, sequence_length, num_heads, head_dim)
        #print(cross_attention_head.shape)
        #print(cross_attention_head)
        cross_attention_head = cross_attention_head.reshape(batch_size, sequence_length, self.num_heads*self.head_dim)
        # Now the Shape of Attention Head will be (batch_size, sequence_length, d_model)
        #print(cross_attention_head.shape)
        #print(cross_attention_head)

        # Inter Communication between Multiple Heads of Cross Attention
        z = self.linear_layer_ca(cross_attention_head)
        # Shape of z tensor will be (batch_size, sequence_length, d_model)
        #print(z.shape)
        return z



# Implementation of Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, hidden, drop_prob = 0.1):
        super(DecoderLayer, self).__init__()
        self.attentiond = MultiheadAttention(d_model=d_model,num_heads=num_heads)
        self.dropout1d = nn.Dropout(p = drop_prob)
        self.norm1d = LayerNormalization(parameters_shape = [d_model])
        self.encoder_decoder_attention = MultiHeadCrossAttention(d_model=d_model,num_heads=num_heads)
        self.dropout2d = nn.Dropout(p = drop_prob)
        self.norm2d = LayerNormalization(parameters_shape = [d_model])
        self.ffnd = PositionwiseFeedForwardNetwork(d_model=d_model,hidden=hidden,drop_prob=drop_prob)
        self.dropout3d = nn.Dropout(p = drop_prob)
        self.norm3d = LayerNormalization(parameters_shape = [d_model])

    def forward(self, x, y, mask1, mask2):
        residual_y = y.clone()
        y = self.attentiond(y, mask1)
        y = self.dropout1d(y)
        y = self.norm1d(y + residual_y)
        residual_y = y.clone()
        y = self.encoder_decoder_attention(x, y, mask2)
        y = self.dropout2d(y)
        y = self.norm2d(y + residual_y)
        residual_y = y.clone()
        y = self.ffnd(y)
        y = self.dropout3d(y)
        y = self.norm3d(y + residual_y)
        return y


class SequentialDecoderLayersGeneration(nn.Sequential):
    def forward(self, x, y, mask1, mask2):
        for j in self._modules.values():
            y = j(x, y, mask1, mask2)
        return y



# Implementation of Decoder
class Decoder(nn.Module):
    def __init__(self,
                d_model,
                num_heads,
                hidden,
                num_layers,
                max_sequence_length,
                language_to_index,
                START_TOKEN,
                END_TOKEN,
                PADDING_TOKEN,
                drop_prob = 0.1):
        super(Decoder, self).__init__()
        self.output_sentence_embedding = SentenceEmbedding(max_sequence_length,d_model,language_to_index,START_TOKEN,END_TOKEN,PADDING_TOKEN)
        self.layersd = SequentialDecoderLayersGeneration(*[DecoderLayer(d_model, num_heads, hidden, drop_prob = drop_prob)
                        for i in range(num_layers)])
    
    def forward(self, x, y, mask1, mask2, decoder_start_token, decoder_end_token):
        y = self.output_sentence_embedding(y, decoder_start_token, decoder_end_token)
        y = self.layersd(x, y, mask1, mask2)
        return y



# Implementation of Transformer Model
class Transformer(nn.Module):
    def __init__(self,
                d_model,
                num_heads,
                hidden,
                num_layers,
                max_sequence_length,
                english_to_index,
                hindi_to_index,
                hindi_vocabulary_size,
                START_TOKEN,
                END_TOKEN,
                PADDING_TOKEN,
                drop_prob = 0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model,num_heads,hidden,num_layers,max_sequence_length,english_to_index,START_TOKEN,END_TOKEN,PADDING_TOKEN,drop_prob = drop_prob)
        self.decoder = Decoder(d_model,num_heads,hidden,num_layers,max_sequence_length,hindi_to_index,START_TOKEN,END_TOKEN,PADDING_TOKEN,drop_prob = drop_prob)
        self.linear = nn.Linear(in_features = d_model, out_features = hindi_vocabulary_size)
    
    def forward(self,
                x,
                y,
                encoder_self_attention_mask,
                decoder_self_attention_mask,
                decoder_cross_attention_mask,
                encoder_start_token,
                encoder_end_token,
                decoder_start_token,
                decoder_end_token) -> Tensor:
        output_encoder = self.encoder(x, encoder_self_attention_mask, encoder_start_token, encoder_end_token)
        output_decoder = self.decoder(output_encoder,y,decoder_self_attention_mask,decoder_cross_attention_mask,decoder_start_token,decoder_end_token)
        output = self.linear(output_decoder)
        return output
    


