# Here we are Generating following three Masks
# encoder_self_attention_mask
# decoder_self_attention_mask
# decoder_cross_attention_mask

import torch

def Generate_Masks(batch, max_sequence_length, NEG_INFTY):
    language_1 = batch[0]
    language_2 = batch[1]
    # num_sentences indicates the batch_size
    num_sentences = len(language_1) 

    look_ahead_mask = torch.full((max_sequence_length, max_sequence_length), True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    encoder_padding_mask = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)
    decoder_padding_mask_self_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)
    decoder_padding_mask_cross_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)

    for i in range(0, num_sentences, 1):
        language_1_length = len(language_1[i])
        language_2_length = len(language_2[i])

        language_1_chars_to_padding_mask = torch.arange(language_1_length, max_sequence_length)
        language_2_chars_to_padding_mask = torch.arange(language_2_length + 2, max_sequence_length)

        encoder_padding_mask[i, :, language_1_chars_to_padding_mask] = True
        encoder_padding_mask[i, language_1_chars_to_padding_mask, :] = True

        decoder_padding_mask_self_attention[i, :, language_2_chars_to_padding_mask] = True
        decoder_padding_mask_self_attention[i, language_2_chars_to_padding_mask, :] = True

        decoder_padding_mask_cross_attention[i, :, language_1_chars_to_padding_mask] = True
        decoder_padding_mask_cross_attention[i, language_2_chars_to_padding_mask, :] = True

    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)
    decoder_self_attention_mask = torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
    decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY, 0)

    return encoder_self_attention_mask,decoder_self_attention_mask,decoder_cross_attention_mask


