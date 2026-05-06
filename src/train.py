import torch
from torch.utils.data import DataLoader
from evaluation import evaluate_batch
from src.losses import squared_emd_loss


# runs the model on a dataset and returns alpha and accuracy without updating weights
def evaluate(model, val_dataset, val_data, collate_fn=None, batch_size=256):
    device = next(model.parameters()).device
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 5:
                encodings, target_tokens_a, target_tokens_b, _, _ = batch
                encodings = {k: v.to(device) for k, v in encodings.items()}
                logits = model(**encodings, target_tokens_a=target_tokens_a, target_tokens_b=target_tokens_b)
            elif len(batch) == 4:
                encodings, target_tokens_a, target_tokens_b, _ = batch
                encodings = {k: v.to(device) for k, v in encodings.items()}
                logits = model(**encodings, target_tokens_a=target_tokens_a, target_tokens_b=target_tokens_b)
            else:
                encodings, _ = batch
                encodings = {k: v.to(device) for k, v in encodings.items()}
                logits = model(**encodings)

            preds = logits.argmax(dim=1).tolist()
            all_preds.extend([p + 1 for p in preds])

    gold = [item["median_label"] for item in val_data]
    return evaluate_batch(all_preds, gold)

def train(model, train_dataset, val_dataset, val_data, is_static=False, tokenizer=None, collate_fn=None, epochs=10, batch_size=32, lr=2e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        collate_fn=collate_fn
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = squared_emd_loss
    best_alpha = -float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            if is_static:
                X_batch, y_batch = batch
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                logits = model(X_batch)

            # a3 target-word pooling — batch has target tokens too
            elif len(batch) == 5:
                encodings, target_tokens_a, target_tokens_b, y_batch, dist_batch = batch
                encodings = {k: v.to(device) for k, v in encodings.items()}
                y_batch = y_batch.to(device)
                dist_batch = dist_batch.to(device)
                optimizer.zero_grad()
                logits = model(**encodings, target_tokens_a=target_tokens_a, target_tokens_b=target_tokens_b)
                loss = criterion(logits, dist_batch)  # bruker distribusjon istedenfor median

            # a2 BOS pooling
            else:
                encodings, y_batch = batch
                encodings = {k: v.to(device) for k, v in encodings.items()}
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                logits = model(**encodings)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in val_loader:
                if is_static:
                    X_batch, _ = batch
                    logits = model(X_batch.to(device))
                # hacky way to detect a3 target-word batches
                elif len(batch) == 5:
                    encodings, target_tokens_a, target_tokens_b, _, _ = batch
                    encodings = {k: v.to(device) for k, v in encodings.items()}
                    logits = model(**encodings, target_tokens_a=target_tokens_a, target_tokens_b=target_tokens_b)
                else:
                    encodings, _ = batch
                    encodings = {k: v.to(device) for k, v in encodings.items()}
                    logits = model(**encodings)

                preds = logits.argmax(dim=1).tolist()
                all_preds.extend([p + 1 for p in preds])

        gold = [item["median_label"] for item in val_data]
        metrics = evaluate_batch(all_preds, gold)
        if metrics["krippendorff_alpha"] > best_alpha:
            best_alpha = metrics["krippendorff_alpha"]
            torch.save(model.state_dict(), "best_model.pt")
            print("Saved new best model")
        print(
            f"Epoch {epoch+1:02d} | "
            f"Loss: {total_loss/len(train_loader):.4f} | "
            f"Alpha: {metrics['krippendorff_alpha']:.4f} | "
            f"Acc: {metrics['accuracy']:.4f}"
        )