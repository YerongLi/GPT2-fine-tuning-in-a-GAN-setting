import numpy as np
import torch 
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class GANGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model_name = "gpt2"  # You can choose a specific GPT-2 variant
       
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


# Function to first select topN tokens from the probability list and then based on the selected N word distribution
    # get random token ID
    def choose_from_top(self, probs, n=5):
        ind = np.argpartition(probs, -n)[-n:]
        top_prob = probs[ind]
        top_prob = top_prob / np.sum(top_prob) # Normalize
        choice = np.random.choice(n, 1, p = top_prob)
        token_id = ind[choice][0]
        return int(token_id)
    
    def generateText(self, input_data , maxLength : int):
        

        out = self.model.generate(**input_data, 
                       max_length = maxLength,
                       do_sample=True,
                        top_k=0,
                        top_p=0.95,
                       num_return_sequences=1,  # Number of generated sequences
                       pad_token_id=50256,)
        return out

    def generate_some_text(self, input_str, training : bool, text_len = 250):

        cur_ids = torch.tensor(self.tokenizer.encode(input_str)).unsqueeze(0).long().to(self.device)
        if training:
            self.model.train()
        else:
            self.model.eval()

        if training == False:
            with torch.no_grad():

                for i in range(text_len):
                    outputs = self.model(cur_ids, labels=cur_ids)
                    loss, logits = outputs[:2]
                    
                    softmax_logits = torch.softmax(logits[0,-1], dim=0) #Take the first(only one) batch and the last predicted embedding
                    next_token_id = self.choose_from_top(softmax_logits.to('cpu').numpy(), n=10) #Randomly(from the given probability distribution) choose the next word from the top n words
                    cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(self.device) * next_token_id], dim = 1) # Add the last word

                output_list = list(cur_ids.squeeze().to('cpu').numpy())
                output_text = self.tokenizer.decode(output_list)
                return output_text
        else:

            for i in range(text_len):
                outputs = self.model(cur_ids, labels=cur_ids)
                loss, logits = outputs[:2]
                
                softmax_logits = torch.softmax(logits[0,-1], dim=0) #Take the first(only one) batch and the last predicted embedding
                next_token_id = self.choose_from_top(softmax_logits.to('cpu').numpy(), n=10) #Randomly(from the given probability distribution) choose the next word from the top n words
                cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(self.device) * next_token_id], dim = 1) # Add the last word

            output_list = list(cur_ids.squeeze().to('cpu').numpy())
            output_text = self.tokenizer.decode(output_list)
            return output_text

        
    def testShapes(self, rawInputs : list):
            
        input_data = self.tokenizer(rawInputs, 
                                    return_tensors="pt", 
                                    padding=True, 
                                    truncation=True, 
                                    max_length=3) # max length refers to the number of words
        outs = self.generateText(input_data=input_data, maxLength=12)
        return outs
            

# gen = GANGenerator()
# testSentences = ["This are the words to continue", 
#                  "hey my name is jenna and i work at starbucks. I Need to work because i need money",
#                   "apples are good",
#                   "I dont like you since the start of july"
                  
#                   ]
# outs = gen.testShapes(rawInputs=testSentences)
# for tokens in outs:
#     print(gen.tokenizer.decode(tokens))