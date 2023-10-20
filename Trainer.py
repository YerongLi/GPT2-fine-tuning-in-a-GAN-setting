import torch
from Discriminator import TextDiscriminatorWithTransformer
from Dataset import GutenbergDataset
from Generator import GANGenerator
from torch.utils.data import DataLoader
from torch.optim import *
from torch.nn import *
"""
The idea is the following: Weve built a discriminator and a generator integrating the GPT2 transformer.

were going to set up a generative adversial training procedure, where instead of optimizing the loss of
of the transformer itself, we will maximize the loss of the discriminator as:

### ORIGINAL LOSS OF THE FIRST GAN PAPER ###

### Discriminator loss ### --> Well use the bce loss

max log(D(Real)) + log(1 - D(G(z)))

with G(z) being the generated data

And the loss of generator as:

### Generator loss ###   --> Well use the bce loss

max log(D(G(z)))  

Once the training is done, well inspect the results of the discriminator on some texts, comparing it to BLUE
or ROUGE
"""


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
lr = 3e-4
batchSize = 32
numEpochs = 50
truncation = 7 # Well cut the input wrt to this parameter and let the generator produce text from there
dataset = GutenbergDataset()
generator = GANGenerator()
discriminator = TextDiscriminatorWithTransformer("gpt2", 1)
optDisc = AdamW(discriminator.parameters(), lr)
optGen = AdamW(generator.parameters(), lr)
lossFunc = torch.nn.BCELoss()
dataloader = DataLoader(dataset=dataset, batch_size= 16, shuffle=True)

# setting up this test training loop

for epoch in range(numEpochs):

    for idx, (batch,real) in enumerate(dataloader):

        ## training the discriminator here
        fakeData = {} # we construct the fake data, and were going to use it twice
        fakeData["attention_mask"] = batch["attention_mask"].squeeze(1)  #The discriminator will know the right attention mask
        batch["input_ids"] =  batch["input_ids"].squeeze(1)[:,:truncation] # truncating the input
        batch["attention_mask"] = batch["attention_mask"].squeeze(1)[:,:truncation]
        discOutsReal = discriminator(batch)  #tensor like, shaped (batchsize, 1)
        fake = generator.generateText(batch, dataset.maxLength) #tensor like, shaped (batchSize, maxLength)
        fakeData["input_ids"] = fake
        discOutsFake = discriminator(fakeData)
        lossDiscriminatorReal = lossFunc(discOutsReal, torch.ones_like(discOutsReal))   # lossFunc(disc(real), torch.oneslike(disc(real)))
        lossDiscriminatorFake = lossFunc(discOutsFake, torch.zeros_like(discOutsFake))
        finalLoss = (lossDiscriminatorReal + lossDiscriminatorFake) / 2
        discriminator.zero_grad()
        finalLoss.backward(retain_graph = True) # adding the retain parameter we ensure that we can use the fake text also for the generator
        optDisc.step()
        
        ## training the generator

        output = discriminator(fakeData) # here the discriminator has been trained once, so this value is different from discOutsFake
        lossGenerator = lossFunc(output, torch.ones_like(output))
        generator.zero_grad()
        lossGenerator.backward()
        optGen.step()

        if idx == 0:
            print("Epoch number ", epoch, " loss Gen: ", lossGenerator, " loss Disc: ", finalLoss)
    
    # once we trained the model for a single epoch, were going to save both models to a local dir

    torch.save(generator.state_dict(), './modelParams/generator' + epoch + ".pth")
    torch.save(discriminator.state_dict(), './modelParams/discriminator' + epoch + ".pth")



    break


