import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import pickle
import matplotlib.pyplot as plt

from torchvision import datasets, transforms

class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        
        self._initialize_weights()

    def _initialize_weights(self):
        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
        
        if self.fc1.bias is not None:
            init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            init.zeros_(self.fc2.bias)
        if self.fc3.bias is not None:
            init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def evaluate(model, device, loader, criterion):
    model.eval()
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(loader.dataset)
    return total_loss / len(loader), accuracy

def train_with_plot(model, device, train_loader, optimizer, criterion, epochs, evaluate_sample_num_per_epoch=None):
    model.train()

    train_size = len(train_loader.dataset)
    mini_batch_size = train_loader.batch_size
    iter_per_epoch = max(train_size // mini_batch_size, 1)
    max_iter = epochs * iter_per_epoch

    print(f"Training for {epochs} epochs, {iter_per_epoch} iterations per epoch, {max_iter} total iterations.")
    
    loss_values = []

    current_iter = 0
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            current_iter += 1
            if current_iter >= max_iter:
                break

        avg_loss = total_loss / iter_per_epoch
        loss_values.append(avg_loss)

        print(f"=== Epoch {epoch+1}/{epochs} Completed ===")

        if evaluate_sample_num_per_epoch:
            eval_loss, eval_accuracy = evaluate(model, device, train_loader, criterion, sample_num=evaluate_sample_num_per_epoch)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Eval Loss: {eval_loss:.4f}, Eval Accuracy: {eval_accuracy:.2f}%")
        else:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), loss_values, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FCNN().to(device)
    train_loader, test_loader = load_data(batch_size=64)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model = train_with_plot(model, device, train_loader, optimizer, criterion, epochs=20)

    with open("fcnn_model.pkl", "wb") as f:
        pickle.dump(model.state_dict(), f)
        print("Saved Model")

    train_loss, train_accuracy = evaluate(model, device, train_loader, criterion)
    test_loss, test_accuracy = evaluate(model, device, test_loader, criterion)

    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")