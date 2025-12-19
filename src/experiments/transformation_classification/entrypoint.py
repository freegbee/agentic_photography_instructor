import logging
from typing import List

from utils.LoggingUtils import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def get_available_transformers() -> List[str]:
    """Get list of available transformer keys from registry."""
    from utils.Registries import TRANSFORMER_REGISTRY, init_registries
    init_registries()
    return sorted(list(TRANSFORMER_REGISTRY.keys()))


def select_transformers(available: List[str], min_count: int = 2) -> List[str]:
    """Interactive transformer selection."""
    print("\nAvailable transformers:")
    for idx, key in enumerate(available):
        transformer = None
        try:
            from utils.Registries import TRANSFORMER_REGISTRY
            transformer = TRANSFORMER_REGISTRY.get(key)
            print(f"  {idx + 1}. {key} - {transformer.description}")
        except Exception as e:
            print(f"  {idx + 1}. {key} - (Error loading: {e})")

    print(f"\nSelect at least {min_count} transformers (comma-separated numbers, e.g., '1,3,5'):")
    try:
        selection = input("Selection: ").strip()
    except EOFError:
        # Default selection for non-interactive mode
        logger.warning("No input available, using default transformers")
        if len(available) >= 2:
            return [available[0], available[1]]
        return available[:min_count]

    if not selection:
        logger.warning("No selection made, using first two transformers")
        return available[:min_count]

    try:
        indices = [int(x.strip()) - 1 for x in selection.split(",")]
        selected = [available[i] for i in indices if 0 <= i < len(available)]

        if len(selected) < min_count:
            logger.error(f"Need at least {min_count} transformers, only got {len(selected)}")
            return available[:min_count]

        return selected
    except (ValueError, IndexError) as e:
        logger.error(f"Invalid selection: {e}")
        return available[:min_count]


def entrypoint():
    """Main entrypoint for transformation classification experiment."""
    print("=" * 70)
    print("Transformation Classification Experiment")
    print("=" * 70)

    # Get experiment parameters
    try:
        experiment_name = input("\nExperiment Name (default: Transformation Classification): ").strip()
    except EOFError:
        experiment_name = ""
    if not experiment_name:
        experiment_name = "Transformation Classification"

    try:
        dataset_id = input("Dataset ID (default: flickr8k): ").strip()
    except EOFError:
        dataset_id = ""
    if not dataset_id:
        dataset_id = "flickr8k"

    try:
        run_name = input("Run name (optional, press Enter to skip): ").strip() or None
    except EOFError:
        run_name = None

    # Select transformers
    available_transformers = get_available_transformers()
    logger.info(f"Found {len(available_transformers)} available transformers")

    selected_transformers = select_transformers(available_transformers, min_count=2)
    logger.info(f"Selected transformers: {selected_transformers}")

    # Training parameters
    try:
        batch_input = input("\nBatch size (default: 32): ").strip()
        batch_size = int(batch_input) if batch_input else 32
    except (EOFError, ValueError):
        batch_size = 32

    try:
        epochs_input = input("Number of epochs (default: 10): ").strip()
        num_epochs = int(epochs_input) if epochs_input else 10
    except (EOFError, ValueError):
        num_epochs = 10

    try:
        lr_input = input("Learning rate (default: 0.001): ").strip()
        learning_rate = float(lr_input) if lr_input else 0.001
    except (EOFError, ValueError):
        learning_rate = 0.001

    print("\n" + "=" * 70)
    print("Configuration Summary:")
    print(f"  Experiment: {experiment_name}")
    print(f"  Run: {run_name or 'auto-generated'}")
    print(f"  Dataset: {dataset_id}")
    print(f"  Transformers: {', '.join(selected_transformers)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print("=" * 70)

    try:
        confirm = input("\nStart experiment? (Y/n): ").strip().lower()
    except EOFError:
        confirm = "y"

    if confirm and confirm != "y":
        print("Experiment cancelled.")
        return

    # Run experiment
    logger.info("Starting experiment with:")
    logger.info(f"  experiment_name={experiment_name}")
    logger.info(f"  run_name={run_name}")
    logger.info(f"  dataset_id={dataset_id}")
    logger.info(f"  transformer_keys={selected_transformers}")
    logger.info(f"  batch_size={batch_size}")
    logger.info(f"  num_epochs={num_epochs}")
    logger.info(f"  learning_rate={learning_rate}")

    from experiments.transformation_classification.TransformationClassificationExperiment import \
        TransformationClassificationExperiment

    exp = TransformationClassificationExperiment(
        experiment_name=experiment_name,
        run_name=run_name,
        dataset_id=dataset_id,
        transformer_keys=selected_transformers,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate
    )

    exp.run()
    print("\n" + "=" * 70)
    print("Experiment completed successfully!")
    print("Check MLflow UI for results: http://localhost:5000")
    print("=" * 70)


if __name__ == "__main__":
    from utils import SslHelper
    SslHelper.create_unverified_ssl_context()

    entrypoint()
