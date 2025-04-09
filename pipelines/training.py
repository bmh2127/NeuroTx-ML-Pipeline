import logging
import os
from pathlib import Path

from common import (
    PYTHON,
    DatasetMixin,
    build_features_transformer,
    build_model,
    build_target_transformer,
    configure_logging,
    packages,
)
from metaflow import (
    FlowSpec,
    Parameter,
    card,
    conda_base,
    current,
    environment,
    project,
    resources,
    step,
)

configure_logging()


@project(name="neurotx")
@conda_base(
    python=PYTHON,
    packages=packages(
        "scikit-learn",
        "pandas",
        "numpy",
        "keras",
        "tensorflow",
        "boto3",
        "mlflow",
        "psutil",
        "pynvml",
        "metaflow",
        "seaborn",
        "deepchem",
        "rdkit",
    ),
)
class NeuroTxPipeline(FlowSpec, DatasetMixin):
    """NeuroTx Pipeline.

    This pipeline processes neuroimaging data, identifies biomarkers, and uses them to
    inform molecular screening criteria. It implements ML models to predict BBB
    permeability and target engagement, and visualizes predicted compound efficacy
    across neural circuits.
    """

    mlflow_tracking_uri = Parameter(
        "mlflow-tracking-uri",
        help="Location of the MLflow tracking server.",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
    )

    training_epochs = Parameter(
        "training-epochs",
        help="Number of epochs that will be used to train the model.",
        default=50,
    )

    training_batch_size = Parameter(
        "training-batch-size",
        help="Batch size that will be used to train the model.",
        default=32,
    )

    accuracy_threshold = Parameter(
        "accuracy-threshold",
        help="Minimum accuracy threshold required to register the model.",
        default=0.7,
    )

    @card
    @step
    def start(self):
        """Start and prepare the NeuroTx pipeline."""
        import mlflow

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        logging.info("MLflow tracking server: %s", self.mlflow_tracking_uri)

        self.mode = "production" if current.is_production else "development"
        logging.info("Running flow in %s mode.", self.mode)

        # Load neuroimaging data
        self.neuro_data = self.load_dataset(dataset_type="neuroimaging")

        # Load molecular data
        self.molecular_data = self.load_dataset(dataset_type="molecular")

        try:
            run = mlflow.start_run(run_name=current.run_id)
            self.mlflow_run_id = run.info.run_id
        except Exception as e:
            message = f"Failed to connect to MLflow server {self.mlflow_tracking_uri}."
            raise RuntimeError(message) from e

        self.next(self.process_neuroimaging, self.process_molecular)

    @card
    @step
    def process_neuroimaging(self):
        """Process neuroimaging data to identify disease-specific biomarkers."""
        logging.info("Processing neuroimaging data...")
        # Implement neuroimaging data processing here
        # Example: self.neuro_features = extract_neuroimaging_features(self.neuro_data)

        self.next(self.identify_biomarkers)

    @card
    @step
    def process_molecular(self):
        """Process molecular data for BBB permeability prediction."""
        logging.info("Processing molecular data...")
        # Implement molecular data processing here
        # Example: self.molecular_features = featurize_molecular_data(self.molecular_data)

        self.next(self.train_molecular_model)

    @card
    @step
    def identify_biomarkers(self):
        """Identify disease-specific biomarkers from neuroimaging data."""
        logging.info("Identifying biomarkers...")
        # Implement biomarker identification here
        # Example: self.biomarkers = identify_biomarkers(self.neuro_features)

        self.next(self.map_biomarkers_to_targets)

    @card
    @step
    def map_biomarkers_to_targets(self):
        """Map identified biomarkers to potential molecular targets."""
        logging.info("Mapping biomarkers to targets...")
        # Implement mapping here
        # Example: self.targets = map_biomarkers_to_targets(self.biomarkers)

        self.next(self.visualize_brain_networks)

    @card
    @step
    def visualize_brain_networks(self):
        """Visualize affected brain networks."""
        logging.info("Visualizing brain networks...")
        # Implement visualization here
        # Example: visualize_brain_networks(self.targets)

        self.next(self.train_molecular_model)

    @card
    @step
    def train_molecular_model(self):
        """Train ML models to predict BBB permeability and target engagement."""
        logging.info("Training molecular model...")
        # Implement model training here
        # Example: self.model = train_bbb_model(self.molecular_features)

        self.next(self.evaluate_model)

    @card
    @step
    def evaluate_model(self):
        """Evaluate the trained molecular model."""
        logging.info("Evaluating model...")
        # Implement model evaluation here
        # Example: self.evaluation_results = evaluate_model(self.model)

        self.next(self.visualize_compound_efficacy)

    @card
    @step
    def visualize_compound_efficacy(self):
        """Visualize predicted compound efficacy across neural circuits."""
        logging.info("Visualizing compound efficacy...")
        # Implement visualization here
        # Example: visualize_compound_efficacy(self.evaluation_results)

        self.next(self.end)

    @step
    def end(self):
        """End the NeuroTx pipeline."""
        logging.info("The pipeline finished successfully.")


if __name__ == "__main__":
    NeuroTxPipeline() 