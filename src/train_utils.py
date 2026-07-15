import torch

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

def train_model(
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    epochs,
    patience,
    save_path
):

    best_val_loss = float("inf")
    counter = 0

    for epoch in range(epochs):

        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in valid_loader:
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                
                pred = model(X_val)
                val_loss += criterion(
                    pred,
                    y_val
                ).item()

        train_loss /= len(train_loader)
        val_loss /= len(valid_loader)

        print(
            f"Epoch {epoch+1}: "
            f"train={train_loss:.6f}, "
            f"val={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(
                model.state_dict(),
                save_path
            )

        else:

            counter += 1
            if counter >= patience:
                print("Early stopping")
                break