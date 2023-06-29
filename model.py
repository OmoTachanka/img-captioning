import torch
import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self, embed_size, train_flag = False):
        super(Encoder, self).__init__()
        self.train_flag = train_flag
        # self.inception = models.inception_v3(pretrained=True)
        # self.inception.aux_logits = False
        # self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, images):
        feats = self.resnet(images)
        for name, param in self.resnet.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = self.train_flag
        return self.dropout(self.relu(feats))

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        # self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim = 0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs
    
class En2De(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super().__init__()
        self.encoder = Encoder(embed_size)
        self.decoder = Decoder(embed_size, hidden_size, vocab_size, num_layers) 
        
    def forward(self, images, captions):
        feats = self.encoder(images)
        ops = self.decoder(feats, captions)
        return ops
    
    def captionize(self, image, vocab, max_length = 50):
        res =[]
        with torch.no_grad():
            x = self.encoder(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoder.lstm(x, states)
                op = self.decoder.linear(hiddens.squeeze(0))
                preds = op.argmax(1)

                res.append(preds.item())
                x = self.decoder.embed(preds).unsqueeze(0)

                if vocab.itos[preds.item()] == '<EOS>':
                    break
        return [vocab.itos[i] for i in res]
        



# if __name__ == "__main__":
#     x = torch.rand(1, 3, 224, 224)
#     model = Encoder(256)
#     print(model(x).shape)
#     print(models.resnet50(weights=ResNet50_Weights.DEFAULT))
