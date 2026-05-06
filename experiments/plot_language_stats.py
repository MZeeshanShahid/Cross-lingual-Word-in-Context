import sys, gzip, json
import matplotlib.pyplot as plt
import numpy as np

DATA_BASE = "/fp/projects01/ec403/IN5550/obligatories/2"

TRAIN_LANGS = ["en", "de", "zh", "ru", "es"]
VAL_LANGS   = ["en", "de", "zh", "ru", "es", "no"]

def get_stats(path):
    lengths, label_counts = [], {1:0, 2:0, 3:0, 4:0}
    with gzip.open(path, "rt") as f:
        for line in f:
            item = json.loads(line)
            lengths.append(len(item["sentence_a"]) + len(item["sentence_b"]))
            label_counts[item["median_label"]] += 1
    return lengths, label_counts

stats = {}
for lang in VAL_LANGS:
    split = "train" if lang in TRAIN_LANGS else "dev"
    path = f"{DATA_BASE}/{lang}_{split}.jsonl.gz"
    try:
        lengths, labels = get_stats(path)
        stats[lang] = {"lengths": lengths, "labels": labels}
    except:
        pass

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

#avg sentence length
langs = list(stats.keys())
avgs = [np.mean(stats[l]["lengths"]) for l in langs]
colors = ["#e74c3c" if l == "no" else "#3498db" for l in langs]
axes[0].bar(langs, avgs, color=colors)
axes[0].set_title("Average sentence pair length (chars)")
axes[0].set_ylabel("Characters")

# dataset size
sizes = [len(stats[l]["lengths"]) for l in langs]
axes[1].bar(langs, sizes, color=colors)
axes[1].set_title("Dataset size")
axes[1].set_ylabel("Number of examples")

# label distribution
x = np.arange(4)
width = 0.13
for i, lang in enumerate(langs):
    total = sum(stats[lang]["labels"].values())
    dist = [stats[lang]["labels"][j+1] / total for j in range(4)]
    color = "#e74c3c" if lang == "no" else None
    axes[2].bar(x + i * width, dist, width, label=lang, color=color)
axes[2].set_xticks(x + width * (len(langs)-1) / 2)
axes[2].set_xticklabels(["1", "2", "3", "4"])
axes[2].set_title("Label distribution")
axes[2].set_ylabel("Proportion")
axes[2].legend()

plt.suptitle("Dataset statistics per language (red = Norwegian)", fontsize=13)
plt.tight_layout()
plt.savefig("results/language_stats.png", dpi=150)
print("Saved to results/language_stats.png")