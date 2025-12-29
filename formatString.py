import pandas as pd
import os
import re

filenames = ["conll00.txt", "conll03.txt", "conll12.txt", "genia.txt", "ontonotes.txt"]



for filename in filenames:
    with open(os.path.join("textFiles", filename), "r", encoding="utf-8") as file:
        lines = file.readlines()
    print(f"Processing {filename}...")

    result = []
    for line in lines:
        match = re.search(r"Precision:\s*([\d.]+),\s*Recall:\s*([\d.]+),\s*F1:\s*([\d.]+)", line, re.IGNORECASE)
        print(line)
        print(match)
        if match:
            precision = float(match.group(1))
            recall = float(match.group(2))
            F1 = float(match.group(3))
            result.append({
                "Precision": precision,
                "Recall": recall,
                "F1": F1
            })
    df = pd.DataFrame(result)
    df.to_csv(os.path.join("textFiles", filename.replace(".txt", ".csv")), index=False)
    print("done")