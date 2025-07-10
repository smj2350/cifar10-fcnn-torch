import torch
import pickle
import matplotlib.pyplot as plt
import sys

sys.path.append('.')
from FCNN_train import FCNN, load_data

def infer(model, device, test_loader):
    """모델을 사용하여 추론 수행"""
    model.eval()
    predictions = []

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            predictions.extend(pred.cpu().numpy())

    return predictions

def visualize_results(model, device, test_loader, num_samples=10):
    """추론 결과를 시각적으로 표현"""
    model.eval()
    x_sample = []
    t_sample = []
    predictions = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            predictions.extend(pred.cpu().numpy())
            x_sample.extend(data.cpu().numpy())
            t_sample.extend(target.cpu().numpy())
            
            if len(x_sample) >= num_samples:
                x_sample = x_sample[:num_samples]
                t_sample = t_sample[:num_samples]
                predictions = predictions[:num_samples]
                break

    for i in range(num_samples):
        plt.figure(figsize=(2, 2))
        plt.imshow(x_sample[i].squeeze(), cmap='gray')
        plt.title(f"Prediction: {predictions[i][0]} (True: {t_sample[i]})")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FCNN().to(device)
    train_loader, test_loader = load_data(batch_size=64)
    
    try:
        with open("fcnn_model.pkl", "rb") as f:
            model.load_state_dict(pickle.load(f))
            print("Model loaded successfully!")
    except FileNotFoundError:
        print("No pre-trained model found. Train the model first.")
        exit()

    visualize_results(model, device, test_loader, num_samples=10)