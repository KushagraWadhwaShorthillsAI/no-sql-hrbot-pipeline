import json
import statistics
from collections import defaultdict
from pathlib import Path

# --- File paths ---
classified_queries_path = "data/classified_queries.json"
manual_keywords_path = "data/manual_keywords.json"
eval_results_path = "data/eval_results.jsonl"
output_summary_path = "logs/classified_eval_summary.txt"
low_precision_json_path = "logs/low_precision_listing_queries.json"

# --- Load classified queries ---
with open(classified_queries_path, "r", encoding="utf-8") as f:
    classified_queries = json.load(f)

# --- Load manual keywords ---
with open(manual_keywords_path, "r", encoding="utf-8") as f:
    manual_keywords = json.load(f)

# --- Load evaluation results ---
with open(eval_results_path, "r", encoding="utf-8") as f:
    eval_results = [json.loads(line.strip()) for line in f if line.strip()]

# --- Collect scores ---
summary = defaultdict(lambda: defaultdict(list))
overall_by_mode = defaultdict(lambda: {"precision": [], "recall": []})
low_precision_entries = []

for result in eval_results:
    query = result["query"]
    mode = result["mode"]
    precision = result["precision"]
    recall = result["recall"]
    retrieved_names = set(name.lower() for name in result.get("retrieved_names", []))
    golden_names = set(name.lower() for name in result.get("golden_names", []))
    noise_names = sorted(retrieved_names - golden_names)
    missed_names = sorted(golden_names - retrieved_names)

    overall_by_mode[mode]["precision"].append(precision)
    overall_by_mode[mode]["recall"].append(recall)

    for class_name, query_set in classified_queries.items():
        if query in query_set:
            summary[class_name][mode].append((precision, recall))

            if class_name == "direct_listing" and precision < 0.6:
                entry = {
                    "query": query,
                    "mode": mode,
                    "precision": precision,
                    "recall": recall,
                    "keywords": manual_keywords.get(query, []),
                    "golden_names": sorted(golden_names),
                    "retrieved_names": sorted(retrieved_names),
                    "noise_names": noise_names,
                    "missed_names": missed_names 
                }
                low_precision_entries.append(entry)

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

# --- Save Summary File ---
Path(output_summary_path).parent.mkdir(exist_ok=True)
with open(output_summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines) + "\n")

# --- Save Low-Precision Listings as JSON ---
if low_precision_entries:
    with open(low_precision_json_path, "w", encoding="utf-8") as f:
        json.dump(low_precision_entries, f, indent=2, ensure_ascii=False)

# --- Print Summary ---
print("\n".join(summary_lines))
