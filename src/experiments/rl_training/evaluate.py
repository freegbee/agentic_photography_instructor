import logging
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from torch.utils.data import DataLoader

from data_types.AgenticImage import ImageData
from experiments.rl_training.RLDataset import RLDataset
from experiments.subset_training.DQNAgent import DQNAgent, ResNetFeatureQNetwork
from experiments.subset_training.TransformationActor import TransformationActor
from transformer import REVERSIBLE_TRANSFORMERS, POC_ONE_WAY_TRANSFORMERS, POC_TWO_WAY_TRANSFORMERS
from utils.ImageUtils import ImageUtils
from utils.LoggingUtils import configure_logging
from dataset.Utils import Utils as DatasetUtils

configure_logging()
logger = logging.getLogger(__name__)


class RLEvaluator:
    """
    Evaluate a trained RL agent on a test dataset.

    Loads a checkpoint and runs the agent on test images,
    logging statistics and optionally saving before/after images.
    """

    def __init__(
        self,
        checkpoint_path: str,
        test_dataset_root: Path,
        max_steps_per_episode: int = 5,
        state_shape: Tuple[int, int, int] = (3, 384, 384),
        action_space: List[str] = None,
        add_stop_action: bool = True,
        save_examples: bool = True,
        output_dir: Path = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.test_dataset_root = test_dataset_root
        self.max_steps_per_episode = max_steps_per_episode
        self.state_shape = state_shape
        self.save_examples = save_examples
        self.output_dir = output_dir or Path("/tmp/rl_evaluation")

        # Action space
        if action_space is None:
            self.action_space = REVERSIBLE_TRANSFORMERS.copy()
            self.action_space = POC_ONE_WAY_TRANSFORMERS.copy()
            self.action_space = POC_TWO_WAY_TRANSFORMERS.copy()
        else:
            self.action_space = action_space

        if add_stop_action:
            self.action_space.append("STOP")

        # Load agent
        network_constructor = ResNetFeatureQNetwork
        network_kwargs = dict(backbone='resnet18', pretrained=True, freeze_backbone=True, use_imagenet_norm=False)
        self.agent = DQNAgent(action_space=self.action_space, state_shape=self.state_shape, network_constructor=network_constructor, network_kwargs=network_kwargs)
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint_info = self.agent.load_checkpoint(checkpoint_path, load_optimizer=False)
        logger.info(f"Loaded checkpoint: {checkpoint_info}")

        # Set to greedy evaluation mode
        self.agent.epsilon = 0.0

        self.transformation_actor = TransformationActor()

        if self.save_examples:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving example outputs to {self.output_dir}")

    def evaluate(self) -> dict:
        """Run evaluation on test dataset."""
        # Load test dataset
        test_images_root = self.test_dataset_root / "test" / "images"
        test_annotations = self.test_dataset_root / "test" / "annotations.json"

        test_dataset = RLDataset(test_images_root, test_annotations)
        dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=DatasetUtils.collate_keep_size)

        logger.info(f"Evaluating on {len(test_dataset)} test samples")

        results = []
        total_improvement = 0.0
        successful_improvements = 0
        total_degradation = 0
        total_steps = 0

        for idx, batch in enumerate(dataloader):
            degraded_image_data, original_score, transformation = batch[0]

            # Run episode
            final_image, final_score, steps_taken, action_sequence = self._run_episode(
                degraded_image_data
            )

            improvement = final_score - degraded_image_data.score
            total_improvement += improvement
            total_steps += steps_taken

            if improvement > 0:
                successful_improvements += 1
            if improvement < 0:
                total_degradation += abs(improvement)

            # Log result
            logger.info(
                f"Sample {idx + 1}/{len(test_dataset)}: "
                f"initial_score={degraded_image_data.score:.4f}, "
                f"final_score={final_score:.4f}, "
                f"improvement={improvement:.4f}, "
                f"steps={steps_taken}, "
                f"actions={action_sequence}"
            )

            results.append({
                "image_id": degraded_image_data.id,
                "image_path": str(degraded_image_data.image_relative_path),
                "initial_score": degraded_image_data.score,
                "final_score": final_score,
                "original_score": original_score,
                "improvement": improvement,
                "steps_taken": steps_taken,
                "actions": action_sequence,
            })

            # Save example outputs
            if self.save_examples and idx < 20:  # Save first 20 examples
                self._save_example(
                    degraded_image_data, final_image, improvement, action_sequence, idx
                )

        # Compute summary statistics
        num_samples = len(results)
        avg_improvement = total_improvement / num_samples
        success_rate = successful_improvements / num_samples
        avg_steps = total_steps / num_samples
        avg_degradation = total_degradation / max(1, num_samples - successful_improvements)

        summary = {
            "num_samples": num_samples,
            "avg_improvement": avg_improvement,
            "success_rate": success_rate,
            "avg_steps": avg_steps,
            "avg_degradation": avg_degradation,
            "total_improvement": total_improvement,
            "successful_improvements": successful_improvements,
        }

        logger.info("=" * 80)
        logger.info("Evaluation Summary")
        logger.info("=" * 80)
        for key, value in summary.items():
            logger.info(f"{key}: {value}")
        logger.info("=" * 80)

        return summary, results

    def _run_episode(self, image_data: ImageData) -> Tuple[np.ndarray, float, int, List[str]]:
        """
        Run one episode with the trained agent.

        Returns:
            (final_image, final_score, steps_taken, action_sequence)
        """
        current_image = image_data.get_image_data("BGR").copy()
        current_score = image_data.score
        action_sequence = []

        for step in range(self.max_steps_per_episode):
            # Preprocess state
            state = self._preprocess_image_to_state(current_image)

            # Select action (greedy)
            action_idx = self.agent.select_action(state)
            action_str = self.agent.action_to_string(action_idx)

            action_sequence.append(action_str)

            # Check for stop action
            if action_str == "STOP":
                break

            # Apply transformation
            try:
                transformed_image, new_score = self.transformation_actor.transform_and_score(
                    current_image, action_str
                )
                logger.info(" Step %d: Image=%s, Action=%s, Score before=%.4f, Score after=%.4f", step, image_data.image_path, action_str, current_score, new_score)
                current_image = transformed_image
                current_score = new_score
            except Exception as e:
                logger.warning(f"Transformation {action_str} failed: {e}")
                break

        return current_image, current_score, len(action_sequence), action_sequence

    def _preprocess_image_to_state(self, img_ndarray: np.ndarray) -> np.ndarray:
        """Convert image to state representation."""
        from PIL import Image
        import cv2

        h_target, w_target = self.state_shape[1], self.state_shape[2]

        if img_ndarray is None:
            return np.zeros(self.state_shape, dtype=np.uint8)

        img_rgb = cv2.cvtColor(img_ndarray, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)
        img = img.resize((w_target, h_target))
        arr = np.asarray(img, dtype=np.uint8)
        arr = np.transpose(arr, (2, 0, 1))
        return arr

    def _save_example(
        self,
        degraded_image_data: ImageData,
        final_image: np.ndarray,
        improvement: float,
        action_sequence: List[str],
        idx: int,
    ):
        """Save before/after comparison for visualization."""
        # Save degraded (initial) image
        degraded_path = self.output_dir / f"example_{idx:03d}_initial.jpg"
        ImageUtils.save_image(degraded_image_data.get_image_data("BGR"), str(degraded_path))

        # Save final image
        final_path = self.output_dir / f"example_{idx:03d}_final.jpg"
        ImageUtils.save_image(final_image, str(final_path))

        # Save metadata
        metadata_path = self.output_dir / f"example_{idx:03d}_metadata.txt"
        with open(metadata_path, "w") as f:
            f.write(f"Image ID: {degraded_image_data.id}\n")
            f.write(f"Image Path: {degraded_image_data.image_relative_path}\n")
            f.write(f"Initial Score: {degraded_image_data.score:.4f}\n")
            f.write(f"Final Score: {degraded_image_data.score + improvement:.4f}\n")
            f.write(f"Improvement: {improvement:.4f}\n")
            f.write(f"Action Sequence: {' -> '.join(action_sequence)}\n")


def main():
    """Command-line entrypoint for evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained RL agent")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", required=True, help="Path to test dataset root")
    parser.add_argument("--max-steps", type=int, default=5, help="Max steps per episode")
    parser.add_argument("--output-dir", default="/tmp/rl_evaluation", help="Output directory for examples")
    parser.add_argument("--no-save-examples", action="store_true", help="Don't save example outputs")

    args = parser.parse_args()

    dataset_root = Path(os.environ.get("IMAGE_VOLUME_PATH", ".")) / args.dataset

    evaluator = RLEvaluator(
        checkpoint_path=args.checkpoint,
        test_dataset_root=dataset_root,
        max_steps_per_episode=args.max_steps,
        save_examples=not args.no_save_examples,
        output_dir=Path(args.output_dir),
    )

    summary, results = evaluator.evaluate()

    # Optionally save results to CSV
    results_csv_path = Path(args.output_dir) / "results.csv"
    logger.info(f"Saving results to {results_csv_path}")

    import csv
    with open(results_csv_path, "w", newline="") as csvfile:
        fieldnames = ["image_id", "image_path", "initial_score", "final_score",
                      "original_score", "improvement", "steps_taken", "actions"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            result["actions"] = " -> ".join(result["actions"])
            writer.writerow(result)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    from utils import SslHelper
    SslHelper.create_unverified_ssl_context()
    main()
