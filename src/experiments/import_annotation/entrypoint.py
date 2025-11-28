import logging
import sys

from utils.LoggingUtils import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def entrypoint():
    """
    Entrypoint for the annotation import experiment.

    Prompts the user for:
    - Experiment name
    - Path to annotations.json file
    - Optional run name
    """
    print("=" * 60)
    print("Annotation Import Experiment")
    print("=" * 60)
    print()

    # Get experiment name
    try:
        experiment_name = input("Experiment Name (default: 'Annotation Import'): ").strip()
    except EOFError:
        experiment_name = ""

    if not experiment_name:
        experiment_name = "Annotation Import"

    # Get annotation file path
    try:
        annotation_file_path = input("Path to annotations.json file (relative to src/ or absolute): ").strip()
    except EOFError:
        print("\nError: No annotation file path provided", file=sys.stderr)
        sys.exit(1)

    if not annotation_file_path:
        print("\nError: Annotation file path is required", file=sys.stderr)
        sys.exit(1)

    # Get optional run name
    try:
        run_name = input("Run name (optional, press Enter to skip): ").strip()
    except EOFError:
        run_name = ""

    if not run_name:
        run_name = None

    print()
    print("-" * 60)
    print("Configuration:")
    print(f"  Experiment Name: {experiment_name}")
    print(f"  Annotation File: {annotation_file_path}")
    print(f"  Run Name: {run_name if run_name else '(auto-generated)'}")
    print("-" * 60)
    print()

    # Confirm before proceeding
    try:
        confirm = input("Proceed with upload? (y/N): ").strip().lower()
    except EOFError:
        confirm = "n"

    if confirm != "y":
        print("Upload cancelled.")
        sys.exit(0)

    # Import here to avoid circular imports and only after user confirmation
    from experiments.import_annotation.AnnotationImportExperiment import AnnotationImportExperiment

    logger.info("Starting annotation import with experiment_name=%s, annotation_file=%s, run_name=%s",
                experiment_name, annotation_file_path, run_name)

    try:
        exp = AnnotationImportExperiment(
            experiment_name=experiment_name,
            annotation_file_path=annotation_file_path,
            run_name=run_name
        )
        exp.run()
        print()
        print("=" * 60)
        print("SUCCESS: Annotation file uploaded to MLflow!")
        print("=" * 60)
    except FileNotFoundError as e:
        print()
        print("=" * 60)
        print(f"ERROR: {e}")
        print("=" * 60)
        print("\nPlease check that the file path is correct.")
        print("Supported path formats:")
        print("  - Relative to src/: data_types/annotations.json")
        print("  - Absolute: /app/src/data_types/annotations.json")
        sys.exit(1)
    except Exception as e:
        print()
        print("=" * 60)
        print(f"ERROR: Failed to upload annotation file")
        print(f"Details: {e}")
        print("=" * 60)
        logger.exception("Exception during annotation import")
        sys.exit(1)


if __name__ == "__main__":
    from utils import SslHelper

    SslHelper.create_unverified_ssl_context()

    entrypoint()
