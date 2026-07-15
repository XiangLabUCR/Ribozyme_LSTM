import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from src.model import LSTMModel
from matplotlib import pyplot as plt

model = LSTMModel(
    input_size=113,
    hidden_size=256,
    output_size=1
)

model.load_state_dict(
    torch.load(
        "models/best_model.pt"
    )
)
model.eval()



# load validation tensors here
data = torch.load("./data/processed/data_split.pt")

X_train = data["X_train"]
y_train = data["y_train"]

X_valid = data["X_valid"]
y_valid = data["y_valid"]


with torch.no_grad():
    predictions = model(X_valid).squeeze().numpy()

mse = mean_squared_error(
    y_valid,
    predictions
)

r2 = r2_score(
    y_valid,
    predictions
)


print("MSE:", mse)
print("R2:", r2)

# plot results
plt.figure()
plt.scatter(y_valid, predictions, c='r', alpha=0.5)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("Predicted vs Actual Values")
plt.savefig('./data/processed/scatterplot.png',dpi=300,bbox_inches='tight')