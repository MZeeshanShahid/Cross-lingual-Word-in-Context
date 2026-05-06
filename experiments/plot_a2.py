import json
import matplotlib.pyplot as plt


# Load the JSON logs for BERT and mmBERT (the JSON files are not included in the GitHub)
with open("/fp/homes01/u01/ec-muhammazs/in5550-research/2/src/a2/ce2_bertbase.json", "r", encoding="utf-8") as f:
    bert = json.load(f)
with open("/fp/homes01/u01/ec-muhammazs/in5550-research/2/src/a2/ce2_mmbert.json", "r", encoding="utf-8") as f:
    mmb = json.load(f)

# Loading epoch-lists
epochs_bert = list(range(1, len(bert["train_loss"]) + 1))
epochs_mmb  = list(range(1, len(mmb["train_loss"]) + 1))

# Plotting training loss
plt.figure()
plt.plot(epochs_bert, bert["train_loss"], label="BERT (bert-base-cased)")
plt.plot(epochs_mmb,  mmb["train_loss"],  label="mmBERT (mmBERT-base)")
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.legend()
plt.tight_layout()
plt.savefig("a2_training_loss_both_2.png", dpi=200)

# Plotting val_alpha
plt.figure()
plt.plot(epochs_bert, bert["val_alpha"], label="BERT (bert-base-cased)")
plt.plot(epochs_mmb,  mmb["val_alpha"],  label="mmBERT (mmBERT-base)")
plt.xlabel("Epoch")
plt.ylabel("Krippendorff's α (English dev)")
plt.legend()
plt.tight_layout()
plt.savefig("a2_val_alpha_both_2.png", dpi=200)

