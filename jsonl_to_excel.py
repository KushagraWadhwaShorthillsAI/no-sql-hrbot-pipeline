import json
import pandas as pd
from pathlib import Path

def jsonl_to_excel(jsonl_path: str, excel_path: str):
    records = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            records.append({
                "Query": record.get("query", ""),
                "Mode": record.get("mode", ""),
                "Precision": record.get("precision", 0),
                "Recall": record.get("recall", 0),
                "Matched Names": ", ".join(record.get("matched_names", [])),
                "Retrieved Names": ", ".join(record.get("retrieved_names", [])),
                "Golden Names": ", ".join(record.get("golden_names", [])),
            })

    df = pd.DataFrame(records)
    df.to_excel(excel_path, index=False)
    print(f"✅ Excel saved to: {excel_path}")

# Example usage:
if __name__ == "__main__":
    jsonl_file = "data/eval_results.jsonl"
    excel_out = "data/eval_results.xlsx"
    jsonl_to_excel(jsonl_file, excel_out)
