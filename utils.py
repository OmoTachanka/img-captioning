import torch
import torchvision.transforms as transforms
from PIL import Image

def print_examples(model, device, dataset):
    transform = transforms.Compose([
        transforms.Resize((299,299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    model.eval()
    
    print()
    test_img1 = transform(Image.open("./test_images/dog.jpg").convert("RGB")).unsqueeze(0)
    print("Example 1 CORRECT: Dog on a beach by the ocean")
    print("Example 1 OP: "+ " ".join(model.captionize(test_img1.to(device), dataset.vocab)))

    test_img2 = transform(Image.open("./test_images/child.jpg").convert("RGB")).unsqueeze(0)
    print("Example 2 CORRECT: Child holding red frisbee outdoors")
    print("Example 2 OP: " + " ".join(model.captionize(test_img2.to(device), dataset.vocab)))

    test_img3 = transform(Image.open("./test_images/bus.png").convert("RGB")).unsqueeze(0)
    print("Example 3 CORRECT: Bus driving by parked cars")
    print("Example 3 OP: " + " ".join(model.captionize(test_img3.to(device), dataset.vocab)))
    
    test_img4 = transform(Image.open("./test_images/boat.png").convert("RGB")).unsqueeze(0)
    print("Example 4 CORRECT: A small boat in the ocean")
    print("Example 4 OP: " + " ".join(model.captionize(test_img4.to(device), dataset.vocab)))

    test_img5 = transform(Image.open("./test_images/horse.png").convert("RGB")).unsqueeze(0)
    print("Example 5 CORRECT: A cowboy riding a horse in the desert")
    print("Example 5 OP: " + " ".join(model.captionize(test_img5.to(device), dataset.vocab)))
    print()

    model.train()

def save_chkpt(state, filename = "chkpt.pth.tar"):
    print("Saving")
    torch.save(state, filename)

def load_chkpt(chkpt, model, optimizer):
    print("Loading")
    model.load_state_dict(chkpt["state_dict"])
    optimizer.load_state_dict(chkpt["optimizer"])
    step = chkpt["step"]
    return step
