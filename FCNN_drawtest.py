import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.widgets import Button
from FCNN_train import FCNN

class DrawingApp:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.img = np.zeros((28, 28))
        self.fig, self.ax = plt.subplots(figsize=(4, 4))
        self.canvas = self.ax.imshow(self.img, cmap='gray', vmin=0, vmax=1)
        self.drawing = False

        self.ax.set_title("Draw for FCNN")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_drag)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

        ax_button_predict = plt.axes([0.7, 0.01, 0.2, 0.05])
        self.button_predict = Button(ax_button_predict, 'Predict')
        self.button_predict.on_clicked(self.predict)

        ax_button_clear = plt.axes([0.1, 0.01, 0.2, 0.05])
        self.button_clear = Button(ax_button_clear, 'Clear')
        self.button_clear.on_clicked(self.clear)

    def on_press(self, event):
        if event.inaxes == self.ax:
            self.drawing = True

    def on_drag(self, event):
        if self.drawing and event.inaxes == self.ax:
            x, y = int(event.xdata), int(event.ydata)
            self.img[max(y - 1, 0):min(y + 2, 28), max(x - 1, 0):min(x + 2, 28)] = 1
            self.canvas.set_data(self.img)
            self.fig.canvas.draw()

    def on_release(self, event):
        self.drawing = False

    def predict(self, event):
        """현재 그림판 이미지를 모델에 전달하여 예측"""
        input_img = torch.tensor(self.img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_img)
            prediction = output.argmax(dim=1).item()

        print(f"Prediction: {prediction}")
        self.ax.set_title(f"Prediction: {prediction}")
        self.fig.canvas.draw()

    def clear(self, event):
        """그림판 초기화"""
        self.img = np.zeros((28, 28))
        self.canvas.set_data(self.img)
        self.ax.set_title("MNIST for FCNN")
        self.fig.canvas.draw()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FCNN().to(device)
    
    try:
        with open("fcnn_model.pkl", "rb") as f:
            model.load_state_dict(pickle.load(f))
            print("Model loaded successfully!")
    except FileNotFoundError:
        print("No pre-trained model found. Train the model first.")
        exit()

    app = DrawingApp(model, device)
    plt.show()
