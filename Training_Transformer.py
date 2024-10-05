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

batch_size = 64
NEG_INFTY = -torch.inf
learning_rate = 0.0001
num_epochs = 30

torch.autograd.set_detect_anomaly(True)

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

# Reading English File and Bring Data in a List
with open('English_Hindi_Data/english.txt', 'r', encoding='utf-8') as file:
    english_sentences = file.readlines()

# Reading Hindi File and Bring Data in a List
with open('English_Hindi_Data/hindi.txt', 'r', encoding='utf-8') as file:
    hindi_sentences = file.readlines()

# Limiting the Total Number of Sentences
TOTAL_SENTENCES = 500000
english_sentences = english_sentences[0:TOTAL_SENTENCES:1]
hindi_sentences = hindi_sentences[0:TOTAL_SENTENCES:1]

# Removing the '\n' After Each English  Sentence
english_sentences = [i.rstrip('\n') for i in english_sentences]
# Removing the '\n' After Each Hindi  Sentence
hindi_sentences = [i.rstrip('\n') for i in hindi_sentences]

# Computing Length of English Sentences
length_english_sentences = [len(i) for i in english_sentences]
# Computing Length of Hindi Sentences
length_hindi_sentences = [len(i) for i in hindi_sentences]

# Applying Percentile Score To Decide Maximum Sequence Length
PERCENTILE = 95
# Computing Percentile Score for English Sentences
y1 = np.percentile(length_english_sentences, PERCENTILE)
# Computing Percentile Score for Hindi Sentences
y2 = np.percentile(length_hindi_sentences, PERCENTILE)
print(y1,y2)

# We Decided Following is the Maximum Sequence Length for English and Hindi
max_sequence_length = 260

# Checking whether the Length of Each Sentence is within Limit
def valid_length_checking(sentence, max_sequence_length):
    if len(sentence) < (max_sequence_length-2):
        return True
    else:
        return False

# Checking whether the Each Character of Each Sentence is within Vocabulary
def valid_tokens_checking(sentence, vocabulary):
    for i in set(sentence):
        if i not in vocabulary:
            return False
    return True

# Selecting only those English and Hindi Sentences such that Valid Length
# and Valid Tokens Conditions satisfy
valid_sentence_index = []
for i in range(len(english_sentences)):
    english_sentence = english_sentences[i]
    hindi_sentence = hindi_sentences[i]
    if valid_length_checking(english_sentence, max_sequence_length)\
        and valid_tokens_checking(english_sentence, english_vocabulary)\
            and valid_length_checking(hindi_sentence, max_sequence_length)\
                and valid_tokens_checking(hindi_sentence, hindi_vocabulary):
        valid_sentence_index.append(i)
print(f'Total Number of Sentences for Each Language:, {len(valid_sentence_index)}')
english_sentences = [english_sentences[i] for i in valid_sentence_index]
hindi_sentences = [hindi_sentences[i] for i in valid_sentence_index]
print('Step One Over \n')

# Preparing the English_Hindi_Dataset
class English_Hindi_Dataset(Dataset):
    def __init__(self, english_sentences, hindi_sentences):
        self.english_sentences = english_sentences
        self.hindi_sentences = hindi_sentences
    
    def __len__(self):
        total_number_of_sentences = len(self.english_sentences)
        return total_number_of_sentences
    
    def __getitem__(self, index):
        data_point = self.english_sentences[index], self.hindi_sentences[index]
        return data_point

text_dataset = English_Hindi_Dataset(english_sentences, hindi_sentences)

# Preparing the Data Loader for our English_Hindi_Dataset
train_loader = DataLoader(dataset = text_dataset, batch_size = batch_size, num_workers = 4, drop_last = True)
print('Step Two Over \n')

# Selecting the Device
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_device()

# Generating the Transformer Model
d_model = 512
num_heads = 8
hidden = 2048
num_layers = 3

hindi_vocabulary_size = len(hindi_to_index)
NEG_INFTY = -torch.inf


model = Transformer_Model.Transformer(d_model,num_heads,hidden,num_layers,max_sequence_length,english_to_index,
                                        hindi_to_index,hindi_vocabulary_size,START_TOKEN,END_TOKEN,PADDING_TOKEN)

for params in model.parameters():
    if params.dim() > 1:
        nn.init.xavier_uniform_(params)

print(model)
print('Step Three Over \n')


criterion = nn.CrossEntropyLoss(ignore_index=hindi_to_index[PADDING_TOKEN],reduction='none').to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, betas=(0.9,0.98))

p = SentenceEmbedding(max_sequence_length,d_model,hindi_to_index,START_TOKEN,END_TOKEN,PADDING_TOKEN)
print('Step Four Over \n')

# Starting of Training
model.train()
model.to(device)


for epoch in range(num_epochs):
    
    for batch_number, batch in enumerate(train_loader):
        (a,b,c) = Transformer_Masks.Generate_Masks(batch,max_sequence_length,NEG_INFTY)
        (encoder_self_attention_mask,decoder_self_attention_mask,decoder_cross_attention_mask)=(a,b,c)

        encoder_self_attention_mask.to(device)
        decoder_self_attention_mask.to(device)
        decoder_cross_attention_mask.to(device)

        x = batch[0]
        y = batch[1]

        labels = p.batch_tokenize(y, start_token=False, end_token=True)

        labels = labels.reshape(batch_size*max_sequence_length).to(device)

        optimizer.zero_grad()

        output = model(x,y,encoder_self_attention_mask,decoder_self_attention_mask,
                       decoder_cross_attention_mask,encoder_start_token = False,
                        encoder_end_token = False,decoder_start_token=True,decoder_end_token=True)

        output = output.reshape(batch_size*max_sequence_length, hindi_vocabulary_size)

        loss_1 = criterion(output, labels).to(device)
        
        print('Shape of Loss_1 is =', loss_1.shape)

        valid_labels = torch.where(labels==hindi_to_index[PADDING_TOKEN], 0, 1)

        loss_2 = (loss_1.sum())/(valid_labels.sum())
        
        loss_2.backward()
        # Gradient Clipping by Value
        # AND
        # Gradient Clipping by Norm
        if (epoch==0):
            nn.utils.clip_grad.clip_grad_value_(model.parameters(), 3.0)
            nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 10.0)
        else:
            nn.utils.clip_grad.clip_grad_value_(model.parameters(), 0.7)
            nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1.0)


        optimizer.step()

        print(f'Batch Number is {batch_number+1} and Epoch Number is {epoch+1}')


# This is for Model Saving
torch.save(model.state_dict(), 'model_3_layers_30_epochs_batch_size_64_lr_0.0001.pth')

