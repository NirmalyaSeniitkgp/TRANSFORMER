import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import Transformer_Model
import Transformer_Masks
from Transformer_Model import SentenceEmbedding


START_TOKEN = '<START>'
PADDING_TOKEN = '<PADDING>'
END_TOKEN = '<END>'

NEG_INFTY = -torch.inf

# Prepare English Vocabulary
english_vocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+',
                    ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':',
                    '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
                    'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                    'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
                    'x', 'y', 'z', '{', '|', '}', '~', PADDING_TOKEN, END_TOKEN]

# Prepare Hindi Vocabulary
hindi_vocabulary = [START_TOKEN, 'ऐ', 'ठ', 'र', 'ॐ', 'ॠ', '॰', 'ऑ', 'ड', 'ऱ', 'ॡ', 'ॱ', 'ऒ',
                    'ढ', 'ल', 'ॲ', 'ओ', 'ण', 'ळ', 'ॳ', 'ऄ', 'औ', 'त', 'ऴ', '।', 'ॴ', 'अ',
                    'क', 'थ', 'व', '॥', 'ॵ', 'आ', 'ख', 'द', 'श', '०', 'ॶ', 'इ', 'ग', 'ध', 'ष',
                    '१', 'ॷ', 'ई', 'घ', 'न', 'स', 'क़', '२', 'ॸ', 'उ', 'ङ', 'ऩ', 'ह', 'ख़', '३',
                    'ॹ', 'ऊ', 'च', 'प', 'ग़', '४', 'ॺ', 'ऋ', 'छ', 'फ', 'ज़', '५', 'ॻ', 'ऌ', 'ज',
                    'ब', 'ड़', '६', 'ॼ', 'ऍ', 'झ', 'भ', 'ऽ', 'ढ़', '७', 'ॽ', 'ऎ', 'ञ', 'म', 'फ़',
                    '८', 'ॾ', 'ए', 'ट', 'य', 'य़', '९', 'ॿ', ' ', 'ऀ', 'ी', 'ँ', 'ु', '॑', 'ं',
                    'ू', '॒', 'ॢ', 'ः', 'ृ', '॓', 'ॣ', 'ॄ', '॔', 'ॅ', 'ॕ', 'ॆ', 'ॖ', 'े', 'ॗ', 'ै',
                    'ॉ', 'ऺ', 'ॊ', 'ऻ', 'ो', '़', 'ौ', '्', 'ा', 'ॎ', 'ि', 'ॏ', '!', '"', '#',
                    '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', '<', '=',
                    '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~',
                    PADDING_TOKEN, END_TOKEN]

# Prepare Dictionary for English Vocabulary
english_to_index = {v:k for k, v in enumerate(english_vocabulary)}
index_to_english = {k:v for k, v in enumerate(english_vocabulary)}
print('*****')
print(len(english_to_index))
print('*****')
# Prepare Dictionary for Hindi Vocabulary
hindi_to_index = {v:k for k, v in enumerate(hindi_vocabulary)}
index_to_hindi = {k:v for k, v in enumerate(hindi_vocabulary)}
print('**********')
print(len(hindi_to_index))
print('**********')


# Generating the Transformer Model
d_model = 512
num_heads = 8
hidden = 2048
num_layers = 3
max_sequence_length = 260
hindi_vocabulary_size = len(hindi_to_index)
NEG_INFTY = -torch.inf


model = Transformer_Model.Transformer(d_model,num_heads,hidden,num_layers,max_sequence_length,english_to_index,
                    hindi_to_index,hindi_vocabulary_size,START_TOKEN,END_TOKEN,PADDING_TOKEN)


# This is for Model Loading

model.load_state_dict(torch.load('/home/idrbt-06/Desktop/PY_TORCH/TRANSFORMER/model_3_layers_30_epochs_batch_size_64_lr_0.0001.pth', map_location=torch.device('cpu')))
model.eval()

# This is for Testing
print('Start of Inference')
# Give Your Input Here (i.e., Input English Sentence Here.)
x = ('Water is very important for life.',)


y = ('',)
for i in range(max_sequence_length-3):
    (a,b,c) = Transformer_Masks.Generate_Masks([x,y],max_sequence_length,NEG_INFTY)
    (encoder_self_attention_mask,decoder_self_attention_mask,decoder_cross_attention_mask)=(a,b,c)

    output_generated = model(x,
                            y, 
                            encoder_self_attention_mask, 
                            decoder_self_attention_mask,
                            decoder_cross_attention_mask,
                            encoder_start_token = False,
                            encoder_end_token = False,
                            decoder_start_token = True,
                            decoder_end_token = False,
                            )
    
    print(i)
    next_token_index = torch.argmax(output_generated[0][i], dim = -1)
    next_token_index = next_token_index.item()
    print(next_token_index)
    next_letter = index_to_hindi[next_token_index]
    y = (y[0]+next_letter,)

    if next_letter == END_TOKEN:
        break

# This is the Output Sentence (i.e., Output Hindi Sentence)
print(y[0])




