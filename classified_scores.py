import json
import statistics
from collections import defaultdict
from pathlib import Path

# --- File paths ---
classified_queries_path = "data/classified_queries.json"
eval_results_path = "data/eval_results.jsonl"
output_summary_path = "logs/classified_eval_summary.txt"

# --- Load classified queries ---
with open(classified_queries_path, "r", encoding="utf-8") as f:
    classified_queries = json.load(f)

# --- Load evaluation results ---
with open(eval_results_path, "r", encoding="utf-8") as f:
    eval_results = [json.loads(line.strip()) for line in f if line.strip()]

# --- Collect scores ---
summary = defaultdict(lambda: defaultdict(list))
overall_by_mode = defaultdict(lambda: {"precision": [], "recall": []})

for result in eval_results:
    query = result["query"]
    mode = result["mode"]
    precision = result["precision"]
    recall = result["recall"]

    overall_by_mode[mode]["precision"].append(precision)
    overall_by_mode[mode]["recall"].append(recall)

    for class_name, query_set in classified_queries.items():
        if query in query_set:
            summary[class_name][mode].append((precision, recall))

# --- Format summary ---
summary_lines = ["\n📊 Class-based Evaluation Summary (All Modes)", "-" * 50]

# --- Overall Mode Scores ---
summary_lines.append("OVERALL")
for mode, scores in overall_by_mode.items():
    avg_p = statistics.mean(scores["precision"]) if scores["precision"] else 0.0
    avg_r = statistics.mean(scores["recall"]) if scores["recall"] else 0.0
    summary_lines.append(f"  {mode.upper():<8} | Precision: {avg_p:.4f} | Recall: {avg_r:.4f}")
summary_lines.append("-" * 50)

# --- Per-Class Summary ---
for class_name, mode_data in summary.items():
    summary_lines.append(f"\nClass: {class_name}")
    for mode, scores in mode_data.items():
        if scores:
            precisions, recalls = zip(*scores)
            avg_p = statistics.mean(precisions)
            avg_r = statistics.mean(recalls)
        else:
            avg_p = avg_r = 0.0
        summary_lines.append(f"  {mode.upper():<8} | Precision: {avg_p:.4f} | Recall: {avg_r:.4f}")

summary_lines.append("-" * 50)

# --- Save & Print ---
Path(output_summary_path).parent.mkdir(exist_ok=True)
with open(output_summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines) + "\n")

print("\n".join(summary_lines))
