import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer

class TextDiscriminatorWithTransformer(nn.Module):
    def __init__(self, transformer_model_name, num_classes):
        super(TextDiscriminatorWithTransformer, self).__init__()
        
        # Load pre-trained transformer model and tokenizer
        self.transformer = GPT2Model.from_pretrained(transformer_model_name)
        # Modify architecture as needed (e.g., adding classification layers)
        self.classifier = nn.Sequential(
            nn.Linear(768, num_classes),  # Modify input size based on the transformer's output dimension
            nn.Sigmoid()  # Sigmoid activation for binary classification
        )
        
    def forward(self, x):
        # Tokenize input text
     #   inputs = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True)
        
        # Obtain transformer embeddings
        outputs = self.transformer(**x)
        
        # Use pooled output or hidden states as input to the classifier
        # Here, we're using the pooled output (CLS token)
        
        last_hidden_state = outputs['last_hidden_state']
        
                # Aggregate the hidden states to a single representation for the whole sentence
        aggregated_hidden_state = last_hidden_state.mean(dim=1)  # You can use other aggregation methods as well
        # Apply classification layers
        out = self.classifier(aggregated_hidden_state)      
        return out

# Example usage:
transformer_model_name = "gpt2"  # Change to the specific pre-trained model you want to use
num_classes = 1  # For binary classification

# Initialize the discriminator with the pre-trained transformer
# discriminator = TextDiscriminatorWithTransformer(transformer_model_name, num_classes)

# sample_text = ["sample" for x in range(64)]
# output = discriminator(sample_text)   # shape is (batch_size, 1) 

# print(output.shape)