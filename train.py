import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from get_loader import get_loader
from utils import save_chkpt, load_chkpt, print_examples
from model import En2De

def train():
    transform = transforms.Compose([
        transforms.Resize((356,356)),
        transforms.RandomCrop((299,299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    train_loader, dataset = get_loader(
        root = "./flickr8k/images",
        annotation_file = "./flickr8k/captions.txt",
        trans = transform,
        num_workers = 2,
    )

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = True

    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 2
    lr = 2e-4
    num_epochs = 10

    writer = SummaryWriter("runs/flickr")
    step = 0

    model = En2De(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr = lr)

    if load_model:
        step = load_chkpt(torch.load("chkpt.pth.tar"), model, optimizer)

    model.train()
    for epoch in range(num_epochs):
        print()
        print_examples(model, device, dataset)
        print()
        if save_model:
            chkpt = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
                "step": step
            }
            save_chkpt(chkpt)
        loop = tqdm(enumerate(train_loader), total = len(train_loader), leave = False)
        for idx, (imgs, captions) in loop:
            imgs = imgs.to(device)
            captions = captions.to(device)
            ops = model(imgs, captions[:-1])
            loss = criterion(ops.reshape(-1, ops.shape[2]), captions.reshape(-1))
            writer.add_scalar("Training_Loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_description(f"T Epoch [{epoch}/{num_epochs}]")
        print(f"Epoch: {epoch+1}/{num_epochs}")

if __name__ == "__main__":
    train()