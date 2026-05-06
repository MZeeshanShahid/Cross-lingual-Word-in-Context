import sys, gzip, json
sys.path.append(".")

DATA_BASE = "/fp/projects01/ec403/IN5550/obligatories/2"

LANGS = ["en", "de", "zh", "ru", "es"]

for lang in LANGS:
    path = f"{DATA_BASE}/{lang}_train.jsonl.gz"
    lengths = []
    with gzip.open(path, "rt") as f:
        for line in f:
            item = json.loads(line)
            lengths.append(len(item["sentence_a"]) + len(item["sentence_b"]))
    avg = sum(lengths) / len(lengths)
    max_len = max(lengths)
    over_1000 = sum(1 for l in lengths if l > 7500)
    print(f"{lang}: avg={avg:.0f} | max={max_len} | over_1000={over_1000} ({100*over_1000/len(lengths):.1f}%) | n={len(lengths)}")