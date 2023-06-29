import os
import pandas as pd
import spacy
import torch
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from PIL import Image
spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary:
    def __init__(self, freq):
       self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
       self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
       self.freq = freq

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]
    
    def build_vocab(self, sentence_list):
        freqs = {}
        idx = 4
        for sent in sentence_list:
            for word in self.tokenize_eng(sent):
                if word not in freqs:
                    freqs[word] = 1
                else:
                    freqs[word] += 1
                
                if freqs[word] == self.freq:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        toked = self.tokenize_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in toked
        ]

class Flickr8k(Dataset):
    def __init__(self, root, captions_file, trans = None, freq = 6):
        self.root = root
        self.df = pd.read_csv(captions_file)
        self.trans = trans

        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        self.vocab = Vocabulary(freq)
        self.vocab.build_vocab(self.captions.tolist())

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        caption = self.captions[idx]
        img = self.imgs[idx]
        x = Image.open(os.path.join(self.root, img))

        if self.trans is not None:
            x = self.trans(x)
        
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return x, torch.tensor(numericalized_caption)

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim = 0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets
    
transform = transforms.Compose([
    transforms.Resize((356,356)),
    transforms.RandomCrop((299,299)),
    transforms.ToTensor(),
])

def get_loader(root, annotation_file, trans, batch_size = 32, num_workers = 8, shuffle = True, pin_memory = True):
    dataset = Flickr8k(root=root, captions_file=annotation_file, trans=trans)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory, collate_fn=MyCollate(pad_idx))
    
    return loader, dataset


def main():
    dataloader = get_loader(root="./flickr8k/Images", annotation_file="./flickr8k/captions.txt", trans=transform)

    for idx, (imgs, captions) in enumerate(dataloader):
        print(imgs.shape, captions.shape)
        break


if __name__ == "__main__":
    main()