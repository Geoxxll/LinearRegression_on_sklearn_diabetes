import torch
from torch import nn
import torch.optim
from data_loader import load_diabetes_tensor
from preprocess import train_val_test_split, standardize
from tqdm.auto import tqdm
from model import LinearRegression
from evaluate import mse, rmse, r2_score, plot_loss, scatter_pred

def main():
    # Define Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 1000
    LR = 0.05
    # HIDDEN_UNITS = 5

    # load data
    X, y, feature_names = load_diabetes_tensor()
    input_shape = len(feature_names)
    # print(f"loaded data:\n X: {X[:3]}\n y: {y[:3]}\n feature names: {feature_names}")

    # Split data and standardize
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
    X_train_s, X_val_s, X_test_s, _ = standardize(X_train, X_val, X_test)
    X_train_s = torch.from_numpy(X_train_s).to(torch.float32)
    X_val_s = torch.from_numpy(X_val_s).to(torch.float32)
    X_test_s = torch.from_numpy(X_test_s).to(torch.float32)
    # print(f"Data Splied:\n X_train: {X_train.shape}\n X_val: {X_val.shape}\n X_test:{X_test.shape}")
    # print(f"Total number of dataEntry: {X_train.shape[0]+X_val.shape[0]+X_test.shape[0]}")

    # dataloader
    train_dataset = torch.utils.data.TensorDataset(X_train_s, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # print(f"In train_loader: {next(iter(train_loader))}")
    test_dataset = torch.utils.data.TensorDataset(X_test_s, y_test)

    # Initiate model
    model = LinearRegression(input_shape=input_shape,
                             output_shape=1)

    # Setup loss and optim function
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(
        lr=LR,
        params=model.parameters()
    )

    # Train Loop
    history = []
    for epoch in range(EPOCHS):
        # print(f"Epoch: {epoch}/{EPOCHS} epochs\n-------------------------")
        total_loss = 0
        for batch, (X, y) in enumerate(train_loader):
            model.train()
            # forward pass
            y_preds = model(X).squeeze(-1)
            # Calculate the loss
            loss = loss_fn(y_preds, y)
            total_loss += loss.item()
            # optimier zero grad
            optimizer.zero_grad()
            # loss backward
            loss.backward()
            optimizer.step()
        # Calculate average batch loss
        train_loss = total_loss / len(train_loader)
        history.append(train_loss)
        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d} Train Loss={train_loss:.4f}")
    
    model.eval()
    with torch.inference_mode():
        y_train_pred = model(X_train_s).squeeze(-1)
        y_val_pred   = model(X_val_s).squeeze(-1)
        y_test_pred  = model(X_test_s).squeeze(-1)
        test_loss = loss_fn(y_test_pred, y_test)
        scatter_pred(y_test.numpy(), y_test_pred.numpy(), out_path="results/pred_vs_true_torch.png")

    for name, y_true, y_pred in [
        ("Train", y_train, y_train_pred),
        ("Val",   y_val,   y_val_pred),
        ("Test",  y_test,  y_test_pred),
    ]:
        print(f"{name:5s}  MSE={mse(y_true, y_pred):.3f}  RMSE={rmse(y_true, y_pred):.3f}  R2={r2_score(y_true, y_pred):.3f}")
    plot_loss(history, out_path="results/loss_curve_torch.png")
    print("----------------------------------------")
    print(f"Trained over {EPOCHS} epochs:\n\tTrain Loss: {train_loss:.4f} Test Loss: {test_loss:.4f}")



if __name__ == "__main__":
    main()