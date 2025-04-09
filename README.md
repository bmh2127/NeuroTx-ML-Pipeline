# NeuroTx-ML: Neuroimaging-Informed Drug Discovery Pipeline for CNS Therapeutics

## Overview
NeuroTx-ML is an end-to-end machine learning pipeline that bridges neuroimaging biomarkers with drug discovery for CNS therapeutics. This project demonstrates the integration of neuroscience domain knowledge with modern ML techniques to identify promising drug candidates for neurological disorders.

## Key Features
- Blood-brain barrier permeability prediction using molecular descriptors
- Integration of neuroimaging biomarkers for target identification
- Multi-modal data processing pipeline using Metaflow
- Comprehensive model evaluation framework with interpretability tools
- Containerized workflow for reproducibility and deployment

## Technical Stack
- **ML Framework**: DeepChem, PyTorch
- **Workflow Management**: Metaflow
- **Data Processing**: RDKit, NumPy, Pandas
- **Visualization**: Matplotlib, Plotly
- **Development**: Cursor IDE with custom rules and notepads

## Project Structure
```
NeuroTx-ML/
├── data/
│   ├── raw/
│   ├── processed/
│   └── neuroimaging/
├── src/
│   ├── data_processing/
│   │   ├── bbb_dataset_processing.py
│   │   └── neuroimaging_processing.py
│   ├── models/
│   │   ├── molecular_property_prediction.py
│   │   └── multimodal_integration.py
│   ├── evaluation/
│   │   └── molecular_model_evaluation.py
│   └── pipeline/
│       └── metaflow_pipeline.py
├── notebooks/
├── tests/
├── docker/
└── README.md
```

## Getting Started
```bash
# Clone repository
git clone https://github.com/bmh2127/neurotx-ml.git
cd neurotx-ml

# Set up environment
conda env create -f environment.yml
conda activate neurotx-ml

# Run example pipeline
python pipelines/bbb_pipeline.py run
```

## Project Goals
This project serves as a portfolio piece demonstrating expertise in:
1. Applying ML to drug discovery challenges
2. Integrating neuroscience domain knowledge with data science
3. Building production-ready ML pipelines
4. Implementing best practices in ML engineering

## Future Directions
- Expand to additional CNS disorders and biomarkers
- Implement active learning for compound screening
- Deploy as a web service for collaborative research
- Integrate with public neuroimaging datasets

## About the Author
This project showcases my unique combination of neuroscience expertise (PhD in Cognitive Neuroscience), technical leadership, and machine learning engineering skills. It demonstrates my ability to bridge scientific domains with practical ML applications - particularly valuable for pharmaceutical and biotech companies developing CNS therapeutics.

## License
MIT