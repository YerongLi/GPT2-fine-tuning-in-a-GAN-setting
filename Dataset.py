import numpy as np
import nltk 
import torch
from nltk.tokenize import sent_tokenize
from nltk.corpus import gutenberg
from torch.utils.data import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
# Get a list of file IDs in the Gutenberg corpus


class GutenbergDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer =GPT2Tokenizer.from_pretrained("gpt2", padding_side = "left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.maxLength = 20
        self.createListOfSentence()

    def createListOfSentence(self) -> list:
        file_ids = gutenberg.fileids()
        self.gutenberDataset = []     # its going to be composed of 18 list of list of sentences from each text
        for title in file_ids:
            currentText = gutenberg.raw(title)
        
            sentence = nltk.sent_tokenize(currentText)
            self.gutenberDataset.append(sentence)
        self.sentencesData = []
        for book in self.gutenberDataset:
            for sentence in book:
                self.sentencesData.append(sentence)
    
        return self.sentencesData  # a single list containing lists of sentences
    

    def prepareTruncatedData(self, maxLength: int):
       # each sentence in the dataset is going to be composed of a fixed number of words, lets set
       # this parameter to 20
        pass

    def decode(self, text):
        return self.tokenizer.decode(text)

    def __len__(self):

        return len(self.sentencesData)
    
    def __getitem__(self, index):
        sentence = self.sentencesData[index]
    
        #Sentences that are longer than max length in terms of words are going to be truncated
        tokenizedSentence = self.tokenizer(sentence, 
                                    return_tensors="pt", 
                                    padding=True, 
                                    truncation=True, 
                                    max_length=self.maxLength)
        
        # If instead the sentence has a minor number of words wrt to max lenght, were padding the sentence
        if tokenizedSentence["input_ids"].shape[1] < self.maxLength:
            fillNumber = self.maxLength -  tokenizedSentence["input_ids"].shape[1]
            concat_input = torch.ones([1,fillNumber], dtype=torch.long) * self.tokenizer.eos_token_id
            concat_attention = torch.zeros([1,fillNumber], dtype=torch.long)

            padded_inputs = torch.cat(concat_input+[tokenizedSentence["input_ids"]], dim = 1)
            padded_attention = torch.cat([tokenizedSentence["attention_mask"], concat_attention], dim = 1)
            
            tokenizedSentence["input_ids"] = padded_inputs
            tokenizedSentence["attention_mask"] = padded_attention

        
                
        return tokenizedSentence, sentence


        




