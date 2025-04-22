import os
import json
import asyncio
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
from src.core.retriever import ResumeRetriever
from src.core.filter_and_rerank import FilterAndRerank
import statistics
from src.core.search_builder import bm25_pipeline

class RetrievalEvaluator:
    def __init__(self, queries_path: str, gold_path: str, output_path: str, top_k: int = None, mode="bm25", use_filter: bool = False):
        self.mode = mode
        self.queries_path = queries_path
        self.gold_path = gold_path
        self.output_path = output_path
        self.top_k = top_k
        self.use_filter = use_filter
        
        self.manual_keywords_path = "data/manual_keywords.json"
        self.null_queries_path = "data/null_queries.json"

        self.manual_keywords = self._load_json(self.manual_keywords_path)
        self.null_queries = {item["query_text"] for item in self._load_json(self.null_queries_path)}

        self.queries = self._load_json(queries_path)
        self.gold_answers = self._load_json(gold_path)
        self.retriever = ResumeRetriever()
        self.filter_rerank = FilterAndRerank() if use_filter else None
        self.log_file = Path("logs/evaluation.log")
        self.log_file.parent.mkdir(exist_ok=True)

    def _load_json(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _log(self, message: str):
        print(message)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def _get_relevant_names(self, query: str) -> set:
        return set(
            item["name"].strip().lower()
            for item in self.gold_answers.get(query, [])
            if item.get("name")
        )

    def _get_retrieved_names(self, docs: List[dict]) -> set:
        return set(
            doc.get("name", "").strip().lower()
            for doc in docs if doc.get("name")
        )

    def evaluate(self, query: str, mode: str, retrieved_docs: List[Dict]) -> dict:
        relevant = self._get_relevant_names(query)
        retrieved = retrieved_docs
        retrieved_names = self._get_retrieved_names(retrieved)

        matched = retrieved_names & relevant

        precision = len(matched) / len(retrieved_names) if retrieved_names else 0
        recall = len(matched) / len(relevant) if relevant else 0

        return {
            "query": query,
            "mode": mode,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "matched_names": list(matched),
            "retrieved_names": list(retrieved_names),
            "golden_names": list(relevant)
        }

    # Hybrid evaluation using manual keywords for BM25, natural query for vector
    # async def run(self):
    #     self._log("\n========== Starting Hybrid Evaluation (BM25, Vector, Fused RRF) ==========")
    #     results = []

    #     for query in self.queries:
    #         if query in self.null_queries:
    #             self._log(f"⏭️  Skipping null query: {query}")
    #             continue

    #         self._log(f"\n🔍 Query: {query}")

    #         keywords = self.manual_keywords.get(query, [])
    #         if not keywords:
    #             self._log("⚠️  No manual keywords found. Skipping.")
    #             continue

    #         # hybrid_search expects: query (str), bm25_override (List[str])
    #         retrieval_output = await self.retriever.hybrid_search(
    #             query=query,                   # vector search uses this
    #             bm25_override=keywords,        # override bm25 with keyword list
    #             top_n_each=300
    #         )

    #         for mode in ["bm25", "vector", "fused"]:
    #             eval_result = self.evaluate(query, mode, retrieval_output[mode])
    #             results.append(eval_result)
    #             self._log(f"  ✅ {mode.upper()} | Precision: {eval_result['precision']} | Recall: {eval_result['recall']}")

    #     # Save results
    #     json_out = Path(self.output_path)
    #     json_out.parent.mkdir(exist_ok=True, parents=True)
    #     with open(json_out, "w", encoding="utf-8") as f:
    #         for item in results:
    #             f.write(json.dumps(item, ensure_ascii=False) + "\n")

    #     self._log(f"\n✅ Hybrid Evaluation complete. Results saved to {self.output_path}")

    async def run(self):
        results = []

        if self.mode == "regex":
            self._log("\n========== Starting Regex Evaluation ==========")
            for query in self.queries:
                if query in self.null_queries:
                    self._log(f"⏭️  Skipping null query: {query}")
                    continue

                self._log(f"\n🔍 Query: {query}")
                keywords = self.manual_keywords.get(query, [])
                if not keywords:
                    self._log("⚠️  No manual keywords found. Skipping.")
                    continue

                mongo_query = self.retriever.build_mongo_query(keywords)
                matched_docs = list(self.retriever.collection.find(mongo_query))

                if self.use_filter:
                    self._log("🧠 Applying filter and rerank...")
                    filtered_result = await self.filter_rerank.filter_and_rerank(query, matched_docs, top_k=self.top_k)
                    matched_docs = filtered_result["filtered"]

                eval_result = self.evaluate(query, "regex", matched_docs)
                results.append(eval_result)
                self._log(f"  ✅ REGEX | Precision: {eval_result['precision']} | Recall: {eval_result['recall']}")

        elif self.mode == "bm25":
            self._log("\n========== Starting BM25 Evaluation ==========")
            for query in self.queries:
                if query in self.null_queries:
                    self._log(f"⏭️  Skipping null query: {query}")
                    continue

                self._log(f"\n🔍 Query: {query}")
                keywords = self.manual_keywords.get(query, [])
                if not keywords:
                    self._log("⚠️  No manual keywords found. Skipping.")    
                    continue

                result = await self.retriever.search(query, override_keywords=keywords)
                bm25_results = result["matched"]

                if self.use_filter:
                    self._log("🧠 Applying filter and rerank...")
                    filtered_result = await self.filter_rerank.filter_and_rerank(query, bm25_results, top_k=self.top_k)
                    bm25_results = filtered_result["filtered"]

                eval_result = self.evaluate(query, "bm25", bm25_results)
                results.append(eval_result)
                self._log(f"  ✅ BM25 | Precision: {eval_result['precision']} | Recall: {eval_result['recall']}")

        elif self.mode == "keyword":
            self._log("\n========== Starting Keyword Match Evaluation ==========")
            for query in self.queries:
                if query in self.null_queries:
                    self._log(f"⏭️  Skipping null query: {query}")
                    continue

                self._log(f"\n🔍 Query: {query}")
                keywords = self.manual_keywords.get(query)
                if not keywords:
                    self._log("⏭️  Skipping null keyword query.")
                    continue

                self._log(f"📌 Using manual keywords: {keywords}")
                matched_docs = self.retriever.keyword_search(keywords, top_k=self.top_k)

                if self.use_filter:
                    self._log("🧠 Applying filter and rerank...")
                    filtered_result = await self.filter_rerank.filter_and_rerank(query, matched_docs, top_k=self.top_k)
                    matched_docs = filtered_result["filtered"]

                eval_result = self.evaluate(query, "keyword", matched_docs)
                results.append(eval_result)
                self._log(f"  ✅ KEYWORD | Precision: {eval_result['precision']:.4f} | Recall: {eval_result['recall']:.4f}")

        elif self.mode == "vector":
            self._log("\n========== Starting Vector Evaluation ==========")
            results = []

            for query in self.queries:
                if query in self.null_queries:
                    self._log(f"⏭️  Skipping null query: {query}")
                    continue

                self._log(f"\n🔍 Query: {query}")
                keywords = self.manual_keywords.get(query, [])
                if not keywords:
                    self._log("⚠️  No manual keywords found. Skipping.")
                    continue

                self._log(f"📌 Using manual keywords for embedding: {keywords}")
                result = await self.retriever.vector_search(query, override_keywords=keywords, k=self.top_k)

                if self.use_filter:
                    self._log("🧠 Applying filter and rerank...")
                    filtered_result = await self.filter_rerank.filter_and_rerank(query, result, top_k=self.top_k)
                    result = filtered_result["filtered"]

                eval_result = self.evaluate(query, "vector", result)
                results.append(eval_result)
                self._log(f"  ✅ VECTOR | Precision: {eval_result['precision']:.4f} | Recall: {eval_result['recall']:.4f}")

        else:
            self._log(f"❌ Unsupported mode: {self.mode}")
            return

        # Save results
        json_out = Path(self.output_path)
        json_out.parent.mkdir(exist_ok=True, parents=True)
        with open(json_out, "w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        self._log(f"\n✅ Evaluation complete. Results saved to {self.output_path}")

    # async def run(self):
    #     self._log("\n========== Starting Regex-Only Evaluation ==========")
    #     results = []

    #     for query in self.queries:
    #         if query in self.null_queries:
    #             self._log(f"⏭️  Skipping null query: {query}")
    #             continue

    #         self._log(f"\n🔍 Query: {query}")

    #         # Inject manual keywords directly
    #         keywords = self.manual_keywords.get(query, [])
    #         if not keywords:
    #             self._log("⚠️  No manual keywords found. Skipping.")
    #             continue

    #         # Build regex-style OR query
    #         mongo_query = self.retriever.build_mongo_query(keywords)
    #         matched_docs = list(self.retriever.collection.find(mongo_query))

    #         eval_result = self.evaluate(query, "regex", matched_docs)
    #         results.append(eval_result)

    #         self._log(f"  ✅ REGEX | Precision: {eval_result['precision']} | Recall: {eval_result['recall']}")

    #     # Save results
    #     json_out = Path(self.output_path)
    #     json_out.parent.mkdir(exist_ok=True, parents=True)
    #     with open(json_out, "w", encoding="utf-8") as f:
    #         for item in results:
    #             f.write(json.dumps(item, ensure_ascii=False) + "\n")

    #     self._log(f"\n✅ Regex Evaluation complete. Results saved to {self.output_path}")

    
    def summarize_results(self, jsonl_path: str, output_txt_path: str = "logs/eval_summary.txt"):

        mode_scores = defaultdict(lambda: {"precision": [], "recall": []})

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                mode = obj["mode"]
                mode_scores[mode]["precision"].append(obj["precision"])
                mode_scores[mode]["recall"].append(obj["recall"])

        summary_lines = ["\n📊 Retrieval Evaluation Summary\n" + "-" * 40]
        for mode in ["bm25", "vector", "fused"]:
            precisions = mode_scores[mode]["precision"]
            recalls = mode_scores[mode]["recall"]
            avg_p = statistics.mean(precisions) if precisions else 0.0
            avg_r = statistics.mean(recalls) if recalls else 0.0
            summary_lines.append(f"{mode.upper():<8} | Precision: {avg_p:.4f} | Recall: {avg_r:.4f}")

        summary_lines.append("-" * 40)
        summary_str = "\n".join(summary_lines)
        print(summary_str)

        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(summary_str + "\n")

        self._log(f"📄 Summary saved to {output_txt_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=str, default="data/test_queries.json")
    parser.add_argument("--gold", type=str, default="data/golden_answers_with_names.json")
    parser.add_argument("--output", type=str, default="data/eval_results.jsonl")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--summary_only", action="store_true", help="Only summarize existing results")
    parser.add_argument("--mode", type=str, default="bm25", choices=["bm25", "regex", "keyword", "vector"], help="Retrieval mode to evaluate")
    parser.add_argument("--use_filter", action="store_true", help="Use filter and rerank before evaluation")

    args = parser.parse_args()

    evaluator = RetrievalEvaluator(
        queries_path=args.queries,
        gold_path=args.gold,
        output_path=args.output,
        top_k=args.top_k,
        mode=args.mode,
        use_filter=args.use_filter
    )

    if args.summary_only:
        evaluator.summarize_results("data/eval_results.jsonl")
    else:
        asyncio.run(evaluator.run())

