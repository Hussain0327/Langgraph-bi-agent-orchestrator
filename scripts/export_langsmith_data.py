"""
Export training data from LangSmith traces for ML routing classifier.

This script:
1. Exports all traces from LangSmith project
2. Extracts query → agents_called pairs
3. Cleans and validates data
4. Splits into train/val/test sets
5. Falls back to synthetic data generation if needed

Usage:
    python scripts/export_langsmith_data.py --min-examples 200 --output models/training_data.json
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from collections import Counter
import hashlib

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langsmith import Client
from src.config import Config
from src.gpt5_wrapper import GPT5Wrapper


class LangSmithDataExporter:
    """Export and prepare training data from LangSmith traces."""

    def __init__(self, project_name: str = None):
        """
        Initialize the exporter.

        Args:
            project_name: LangSmith project name (defaults to Config.LANGCHAIN_PROJECT)
        """
        self.project_name = project_name or Config.LANGCHAIN_PROJECT
        self.client = Client()
        self.gpt5 = GPT5Wrapper()
        self.valid_agents = ["market", "operations", "financial", "leadgen"]

    def export_traces(self, days_back: int = 30) -> List[Dict[str, Any]]:
        """
        Export all traces from LangSmith.

        Args:
            days_back: Number of days of history to export

        Returns:
            List of trace dictionaries
        """
        print(f"\n📡 Exporting traces from LangSmith project: {self.project_name}")
        print(f"   Looking back {days_back} days...")

        traces = []
        start_time = datetime.now() - timedelta(days=days_back)

        try:
            # Get runs from the project
            runs = self.client.list_runs(
                project_name=self.project_name,
                start_time=start_time,
                execution_order=1  # Only get top-level runs
            )

            for run in runs:
                # Extract relevant information
                trace_data = {
                    "run_id": str(run.id),
                    "name": run.name,
                    "inputs": run.inputs,
                    "outputs": run.outputs,
                    "start_time": run.start_time.isoformat() if run.start_time else None,
                    "end_time": run.end_time.isoformat() if run.end_time else None,
                    "error": run.error,
                    "tags": run.tags or [],
                    "metadata": run.extra or {}
                }
                traces.append(trace_data)

            print(f"✓ Exported {len(traces)} traces from LangSmith")
            return traces

        except Exception as e:
            print(f"⚠️  Error exporting from LangSmith: {e}")
            print(f"   This might be expected if the project is new or has no runs yet.")
            return []

    def extract_training_examples(self, traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract query → agents pairs from traces.

        Args:
            traces: List of trace dictionaries

        Returns:
            List of training examples with format:
            {"query": str, "agents": List[str], "timestamp": str}
        """
        print(f"\n🔍 Extracting training examples from traces...")

        training_examples = []

        for trace in traces:
            try:
                # Try to extract query and agents from inputs/outputs
                query = None
                agents = []

                # Look for query in inputs
                if isinstance(trace.get("inputs"), dict):
                    query = (
                        trace["inputs"].get("query") or
                        trace["inputs"].get("input") or
                        trace["inputs"].get("question")
                    )

                    # Also check if agents_to_call is in inputs (initial state)
                    agents = trace["inputs"].get("agents_to_call", [])

                # Look for agents in outputs
                if not agents and isinstance(trace.get("outputs"), dict):
                    agents = (
                        trace["outputs"].get("agents_to_call", []) or
                        trace["outputs"].get("agents", [])
                    )

                # Validate and add example
                if query and agents:
                    # Validate agents
                    valid_agents = [a for a in agents if a in self.valid_agents]

                    if valid_agents:
                        example = {
                            "query": str(query),
                            "agents": valid_agents,
                            "timestamp": trace.get("start_time", datetime.now().isoformat())
                        }
                        training_examples.append(example)

            except Exception as e:
                # Skip traces that don't match expected format
                continue

        print(f"✓ Extracted {len(training_examples)} valid training examples")

        # Show agent distribution
        if training_examples:
            agent_counts = Counter()
            for example in training_examples:
                for agent in example["agents"]:
                    agent_counts[agent] += 1

            print(f"\n   Agent distribution:")
            for agent, count in agent_counts.most_common():
                print(f"   - {agent}: {count}")

        return training_examples

    def generate_synthetic_examples(self, num_examples: int = 200) -> List[Dict[str, Any]]:
        """
        Generate synthetic training examples using GPT-5.

        Args:
            num_examples: Number of examples to generate

        Returns:
            List of synthetic training examples
        """
        print(f"\n🤖 Generating {num_examples} synthetic training examples...")

        # Seed examples for each agent combination
        seed_queries = {
            ("market",): [
                "What's the market size for B2B SaaS?",
                "Who are the main competitors in e-commerce?",
                "What customer segments should we target?",
            ],
            ("operations",): [
                "How can we optimize our fulfillment process?",
                "What are best practices for customer onboarding?",
                "How do we reduce support ticket volume?",
            ],
            ("financial",): [
                "What's the ROI for this marketing campaign?",
                "How should we price our new feature?",
                "What's our customer acquisition cost?",
            ],
            ("leadgen",): [
                "How can we generate more qualified leads?",
                "What's the best outreach strategy for cold emails?",
                "How do we improve our sales funnel conversion?",
            ],
            ("market", "operations"): [
                "How can we improve customer retention?",
                "What's causing customer churn?",
                "How do we expand into new market segments?",
            ],
            ("market", "financial"): [
                "Should we raise prices for our premium tier?",
                "What's the market opportunity for our new product?",
                "How do we maximize revenue per customer?",
            ],
            ("operations", "financial"): [
                "How can we reduce operational costs?",
                "What's the payback period for this automation?",
                "Should we outsource customer support?",
            ],
            ("market", "leadgen"): [
                "Who should we target for our product launch?",
                "How do we reach enterprise customers?",
                "What messaging resonates with our target market?",
            ],
            ("market", "operations", "financial"): [
                "How do we scale our SaaS business?",
                "What's the best growth strategy for our startup?",
                "How can we improve unit economics?",
            ],
            ("market", "operations", "leadgen"): [
                "How do we build a sales engine for our product?",
                "What's a complete go-to-market strategy?",
                "How do we acquire and retain customers efficiently?",
            ],
        }

        synthetic_examples = []
        target_per_combo = num_examples // len(seed_queries)

        for agent_combo, queries in seed_queries.items():
            print(f"   Generating variations for {agent_combo}...")

            for seed_query in queries:
                # Generate variations of each seed query
                variations_prompt = f"""Generate 5 different phrasings of this business question that mean the same thing:

Original: "{seed_query}"

Requirements:
- Keep the same core business intent
- Use different wording and sentence structure
- Make them sound natural (not templated)
- Return only the questions, one per line

Questions:"""

                try:
                    response = self.gpt5.generate(
                        input_text=variations_prompt,
                        max_output_tokens=500,
                        reasoning_effort="low"
                    )

                    # Parse variations
                    variations = [
                        line.strip().strip('123456789.-) ')
                        for line in response.split('\n')
                        if line.strip() and len(line.strip()) > 10
                    ]

                    # Add original + variations
                    all_queries = [seed_query] + variations[:4]  # Limit to 5 total

                    for query in all_queries:
                        example = {
                            "query": query,
                            "agents": list(agent_combo),
                            "timestamp": datetime.now().isoformat(),
                            "synthetic": True
                        }
                        synthetic_examples.append(example)

                        if len(synthetic_examples) >= num_examples:
                            break

                    if len(synthetic_examples) >= num_examples:
                        break

                except Exception as e:
                    print(f"⚠️  Error generating variations: {e}")
                    # Fallback: just use the seed query
                    example = {
                        "query": seed_query,
                        "agents": list(agent_combo),
                        "timestamp": datetime.now().isoformat(),
                        "synthetic": True
                    }
                    synthetic_examples.append(example)

            if len(synthetic_examples) >= num_examples:
                break

        print(f"✓ Generated {len(synthetic_examples)} synthetic examples")
        return synthetic_examples[:num_examples]

    def clean_and_validate(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean and validate training examples.

        Args:
            examples: Raw training examples

        Returns:
            Cleaned and validated examples
        """
        print(f"\n🧹 Cleaning and validating {len(examples)} examples...")

        cleaned = []
        seen_queries = set()

        for example in examples:
            # Normalize query
            query = example["query"].strip()

            # Skip duplicates
            query_hash = hashlib.md5(query.lower().encode()).hexdigest()
            if query_hash in seen_queries:
                continue
            seen_queries.add(query_hash)

            # Validate agents
            agents = example["agents"]
            if not agents or not isinstance(agents, list):
                continue

            # Filter to valid agents only
            valid_agents = [a for a in agents if a in self.valid_agents]
            if not valid_agents:
                continue

            # Sort agents for consistency
            valid_agents = sorted(set(valid_agents))

            cleaned_example = {
                "query": query,
                "agents": valid_agents,
                "timestamp": example.get("timestamp", datetime.now().isoformat()),
                "synthetic": example.get("synthetic", False)
            }
            cleaned.append(cleaned_example)

        print(f"✓ Cleaned to {len(cleaned)} unique, valid examples")
        return cleaned

    def split_data(
        self,
        examples: List[Dict[str, Any]],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Split data into train/val/test sets.

        Args:
            examples: Training examples
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set

        Returns:
            Dictionary with keys: "train", "val", "test"
        """
        print(f"\n📊 Splitting data: {train_ratio}/{val_ratio}/{test_ratio}...")

        # Shuffle deterministically
        examples_sorted = sorted(examples, key=lambda x: hashlib.md5(x["query"].encode()).hexdigest())

        n = len(examples_sorted)
        train_idx = int(n * train_ratio)
        val_idx = int(n * (train_ratio + val_ratio))

        splits = {
            "train": examples_sorted[:train_idx],
            "val": examples_sorted[train_idx:val_idx],
            "test": examples_sorted[val_idx:]
        }

        print(f"✓ Split: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

        return splits

    def export_dataset(
        self,
        min_examples: int = 200,
        output_path: str = "models/training_data.json"
    ) -> Dict[str, Any]:
        """
        Export complete training dataset.

        Args:
            min_examples: Minimum number of examples required
            output_path: Path to save the dataset

        Returns:
            Dataset dictionary with metadata
        """
        print(f"\n" + "="*70)
        print(f"🚀 LANGSMITH TRAINING DATA EXPORT")
        print(f"="*70)

        # Step 1: Export from LangSmith
        traces = self.export_traces(days_back=30)
        real_examples = self.extract_training_examples(traces)

        # Step 2: Generate synthetic if needed
        all_examples = real_examples
        if len(real_examples) < min_examples:
            print(f"\n⚠️  Only {len(real_examples)} real examples (need {min_examples})")
            print(f"   Generating synthetic examples to supplement...")

            synthetic_needed = min_examples - len(real_examples)
            synthetic_examples = self.generate_synthetic_examples(synthetic_needed)
            all_examples = real_examples + synthetic_examples

        # Step 3: Clean and validate
        cleaned_examples = self.clean_and_validate(all_examples)

        # Step 4: Split data
        splits = self.split_data(cleaned_examples)

        # Step 5: Create dataset
        dataset = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "project_name": self.project_name,
                "total_examples": len(cleaned_examples),
                "real_examples": len(real_examples),
                "synthetic_examples": len(cleaned_examples) - len(real_examples),
                "splits": {
                    "train": len(splits["train"]),
                    "val": len(splits["val"]),
                    "test": len(splits["test"])
                },
                "valid_agents": self.valid_agents
            },
            "train": splits["train"],
            "val": splits["val"],
            "test": splits["test"]
        }

        # Step 6: Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)

        print(f"\n" + "="*70)
        print(f"✅ DATASET EXPORT COMPLETE")
        print(f"="*70)
        print(f"\n📁 Saved to: {output_path}")
        print(f"\n📊 Summary:")
        print(f"   Total examples: {dataset['metadata']['total_examples']}")
        print(f"   - Real: {dataset['metadata']['real_examples']}")
        print(f"   - Synthetic: {dataset['metadata']['synthetic_examples']}")
        print(f"\n   Train: {len(splits['train'])}")
        print(f"   Val:   {len(splits['val'])}")
        print(f"   Test:  {len(splits['test'])}")
        print(f"\n✓ Ready for ML training!")
        print(f"="*70 + "\n")

        return dataset


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Export LangSmith training data for ML routing")
    parser.add_argument(
        "--min-examples",
        type=int,
        default=200,
        help="Minimum number of training examples (will generate synthetic if needed)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/training_data.json",
        help="Output path for training data"
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="LangSmith project name (defaults to env var)"
    )

    args = parser.parse_args()

    # Create exporter and run
    exporter = LangSmithDataExporter(project_name=args.project)
    dataset = exporter.export_dataset(
        min_examples=args.min_examples,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
