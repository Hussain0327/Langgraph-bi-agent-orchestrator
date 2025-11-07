"""ML routing classifier using SetFit for fast, cheap agent selection."""

import json
import logging
import os
import pickle
from typing import List, Dict, Any, Tuple
import numpy as np

from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import Dataset as HFDataset
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RoutingClassifier:
    AGENTS = ["financial", "leadgen", "market", "operations"]

    def __init__(self, base_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.base_model_name = base_model
        self.agent_labels = self.AGENTS
        self.mlb = MultiLabelBinarizer(classes=self.agent_labels)
        self.mlb.fit([self.agent_labels])
        self.models = {}
        self.training_metrics = {}

    def load_training_data(self, data_path: str = "models/training_data.json") -> Tuple[List[str], List[List[str]]]:
        try:
            with open(data_path, 'r') as f:
                dataset = json.load(f)

            train_data = dataset.get("train", [])
            if not train_data:
                raise ValueError(f"No training data found in {data_path}")

            queries = [ex["query"] for ex in train_data]
            labels = [ex["agents"] for ex in train_data]

            logger.info(f"Loaded {len(queries)} training examples from {data_path}")

            # Log label distribution
            label_counts = {}
            for agent_list in labels:
                for agent in agent_list:
                    label_counts[agent] = label_counts.get(agent, 0) + 1

            for agent in sorted(label_counts.keys()):
                logger.info(f"  {agent}: {label_counts[agent]} examples")

            return queries, labels

        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            raise


    def train(self, data_path="models/training_data.json", num_epochs=10,
              batch_size=16, learning_rate=2e-5, save_path="models/routing_classifier.pkl"):
        print(f"\n{'='*70}\n🚀 TRAINING ML ROUTING CLASSIFIER\n{'='*70}")

        queries, labels = self.load_training_data(data_path)

        with open(data_path, 'r') as f:
            dataset = json.load(f)
        val_data = dataset.get("val", [])
        val_queries = [ex["query"] for ex in val_data]
        val_labels = [ex["agents"] for ex in val_data]

        self.models = {}
        agent_metrics = {}

        for agent in self.agent_labels:
            print(f"\n{'─'*70}\n🔧 Training classifier for: {agent}\n{'─'*70}")

            train_binary = [1 if agent in label_list else 0 for label_list in labels]
            val_binary = [1 if agent in label_list else 0 for label_list in val_labels]

            model = SetFitModel.from_pretrained(self.base_model_name, labels=["no", "yes"])
            train_dataset = HFDataset.from_dict({"text": queries, "label": train_binary})
            args = TrainingArguments(batch_size=batch_size, num_epochs=num_epochs)
            trainer = Trainer(model=model, args=args, train_dataset=train_dataset)

            print(f"   Training with {len(train_dataset)} examples...")
            trainer.train()

            val_preds = model.predict(val_queries)
            val_preds_binary = [1 if pred == "yes" else 0 for pred in val_preds]

            precision = precision_score(val_binary, val_preds_binary, zero_division=0)
            recall = recall_score(val_binary, val_preds_binary, zero_division=0)
            f1 = f1_score(val_binary, val_preds_binary, zero_division=0)
            accuracy = accuracy_score(val_binary, val_preds_binary)

            agent_metrics[agent] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "accuracy": float(accuracy),
                "train_examples": sum(train_binary),
                "val_examples": sum(val_binary)
            }

            print(f"   ✓ Validation Metrics:")
            print(f"      Precision: {precision:.3f}")
            print(f"      Recall:    {recall:.3f}")
            print(f"      F1 Score:  {f1:.3f}")
            print(f"      Accuracy:  {accuracy:.3f}")

            self.models[agent] = model

        print(f"\n{'='*70}\n📊 OVERALL TRAINING RESULTS\n{'='*70}")

        avg_f1 = np.mean([m["f1"] for m in agent_metrics.values()])
        avg_precision = np.mean([m["precision"] for m in agent_metrics.values()])
        avg_recall = np.mean([m["recall"] for m in agent_metrics.values()])

        print(f"\nAverage across all agents:")
        print(f"   Precision: {avg_precision:.3f}")
        print(f"   Recall:    {avg_recall:.3f}")
        print(f"   F1 Score:  {avg_f1:.3f}")

        val_predictions = self.predict_batch(val_queries)
        exact_matches = sum(1 for pred, true in zip(val_predictions, val_labels) if set(pred) == set(true))
        exact_match_accuracy = exact_matches / len(val_labels)

        print(f"\nValidation Set Exact Match Accuracy: {exact_match_accuracy:.3f}")
        print(f"   ({exact_matches}/{len(val_labels)} exact matches)")

        self.training_metrics = {
            "timestamp": datetime.now().isoformat(),
            "base_model": self.base_model_name,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "agent_metrics": agent_metrics,
            "avg_f1": float(avg_f1),
            "avg_precision": float(avg_precision),
            "avg_recall": float(avg_recall),
            "exact_match_accuracy": float(exact_match_accuracy),
            "training_examples": len(queries),
            "validation_examples": len(val_queries)
        }
        self.save(save_path)

        print(f"\n✅ Training complete!")
        print(f"="*70 + "\n")

        return self.training_metrics

    def predict(self, query: str, threshold: float = 0.5) -> List[str]:
        if not self.models:
            raise ValueError("Model not trained. Call train() or load() first.")

        agents = []
        try:
            for agent, model in self.models.items():
                prediction = model.predict([query])[0]
                if prediction == "yes":
                    agents.append(agent)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

        # Fallback to market if no agents selected
        return sorted(agents) if agents else ["market"]

    def predict_batch(self, queries: List[str]) -> List[List[str]]:
        return [self.predict(q) for q in queries]

    def predict_proba(self, query: str) -> Dict[str, float]:
        if not self.models:
            raise ValueError("Model not trained. Call train() or load() first.")

        proba = {}
        try:
            for agent, model in self.models.items():
                prediction = model.predict([query])[0]
                proba[agent] = 1.0 if prediction == "yes" else 0.0
        except Exception as e:
            logger.error(f"Probability prediction failed: {e}")
            raise

        return proba

    def evaluate(self, data_path="models/training_data.json"):
        print(f"\n📊 Evaluating on test set...")

        with open(data_path, 'r') as f:
            dataset = json.load(f)

        test_data = dataset.get("test", [])
        test_queries = [ex["query"] for ex in test_data]
        test_labels = [ex["agents"] for ex in test_data]

        predictions = self.predict_batch(test_queries)
        exact_matches = sum(1 for pred, true in zip(predictions, test_labels) if set(pred) == set(true))
        exact_match_accuracy = exact_matches / len(test_labels)

        agent_metrics = {}
        for agent in self.agent_labels:
            true_binary = [1 if agent in labels else 0 for labels in test_labels]
            pred_binary = [1 if agent in preds else 0 for preds in predictions]

            agent_metrics[agent] = {
                "precision": float(precision_score(true_binary, pred_binary, zero_division=0)),
                "recall": float(recall_score(true_binary, pred_binary, zero_division=0)),
                "f1": float(f1_score(true_binary, pred_binary, zero_division=0)),
                "accuracy": float(accuracy_score(true_binary, pred_binary))
            }

        avg_f1 = np.mean([m["f1"] for m in agent_metrics.values()])

        metrics = {
            "test_examples": len(test_queries),
            "exact_match_accuracy": float(exact_match_accuracy),
            "exact_matches": exact_matches,
            "agent_metrics": agent_metrics,
            "avg_f1": float(avg_f1)
        }

        print(f"✓ Test Set Results:")
        print(f"   Exact Match Accuracy: {exact_match_accuracy:.3f}")
        print(f"   Average F1 Score: {avg_f1:.3f}")

        return metrics

    def save(self, path: str = "models/routing_classifier.pkl"):
        if not self.models:
            raise ValueError("No model to save. Train first.")

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)

            save_data = {
                "models": self.models,
                "agent_labels": self.agent_labels,
                "base_model_name": self.base_model_name,
                "training_metrics": self.training_metrics
            }

            with open(path, 'wb') as f:
                pickle.dump(save_data, f)

            # Save metrics separately as JSON
            metrics_path = path.replace(".pkl", "_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(self.training_metrics, f, indent=2)

            logger.info(f"Model saved to {path}")
            logger.info(f"Metrics saved to {metrics_path}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def load(self, path: str = "models/routing_classifier.pkl"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")

        try:
            with open(path, 'rb') as f:
                save_data = pickle.load(f)

            self.models = save_data["models"]
            self.agent_labels = save_data["agent_labels"]
            self.base_model_name = save_data["base_model_name"]
            self.training_metrics = save_data.get("training_metrics", {})

            acc = self.training_metrics.get('exact_match_accuracy', 0)
            logger.info(f"Model loaded from {path} (accuracy: {acc:.3f})")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train ML routing classifier")
    parser.add_argument(
        "--data",
        type=str,
        default="models/training_data.json",
        help="Path to training data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/routing_classifier.pkl",
        help="Path to save trained model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size"
    )

    args = parser.parse_args()

    # Create and train classifier
    classifier = RoutingClassifier()
    metrics = classifier.train(
        data_path=args.data,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        save_path=args.output
    )

    # Evaluate on test set
    test_metrics = classifier.evaluate(data_path=args.data)

    print(f"\n" + "="*70)
    print(f"✅ TRAINING AND EVALUATION COMPLETE")
    print(f"="*70)
    print(f"\nFinal Test Set Metrics:")
    print(f"   Exact Match Accuracy: {test_metrics['exact_match_accuracy']:.3f}")
    print(f"   Average F1 Score: {test_metrics['avg_f1']:.3f}")
    print(f"="*70 + "\n")


if __name__ == "__main__":
    main()
