# models/templates/bbb_model_template.py

import deepchem as dc
from rdkit import Chem
import numpy as np

class BBBPermeabilityModel:
    """
    Template for blood-brain barrier permeability prediction models.

    This template follows the NeuroTx-ML standard structure for
    molecular property prediction models.
    """

    def __init__(self, model_type='graph_conv', featurizer_type='circular'):
        """
        Initialize the BBB permeability prediction model.

        Parameters
        ----------
        model_type : str
            Type of model to use ('graph_conv', 'mpnn', etc.)
        featurizer_type : str
            Type of featurizer to use ('circular', 'graph', etc.)
        """
        self.model_type = model_type

        # Set up featurizer
        if featurizer_type == 'circular':
            self.featurizer = dc.feat.CircularFingerprint(size=1024)
        elif featurizer_type == 'graph':
            self.featurizer = dc.feat.MolGraphConvFeaturizer()
        else:
            raise ValueError(f"Unsupported featurizer type: {featurizer_type}")

        # Initialize model
        if model_type == 'graph_conv':
            self.model = dc.models.GraphConvModel(n_tasks=1, mode='classification')
        elif model_type == 'mpnn':
            self.model = dc.models.MPNNModel(n_tasks=1, mode='classification')
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def featurize_smiles(self, smiles_list):
        """
        Convert SMILES strings to molecular features.

        Parameters
        ----------
        smiles_list : list
            List of SMILES strings

        Returns
        -------
        features : np.ndarray
            Featurized molecules
        """
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        return self.featurizer.featurize(mols)

    def train(self, X, y, n_epochs=100, batch_size=32):
        """
        Train the BBB permeability prediction model.

        Parameters
        ----------
        X : np.ndarray or list
            Features or SMILES strings
        y : np.ndarray
            Binary labels (1 for BBB+, 0 for BBB-)
        n_epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training

        Returns
        -------
        history : dict
            Training history
        """
        # Convert SMILES to features if needed
        if isinstance(X[0], str):
            X = self.featurize_smiles(X)

        # Create dataset
        dataset = dc.data.NumpyDataset(X, y)

        # Split dataset
        splitter = dc.splits.RandomSplitter()
        train_dataset, valid_dataset = splitter.train_test_split(dataset, frac_train=0.8)

        # Train model
        return self.model.fit(train_dataset, nb_epoch=n_epochs, batch_size=batch_size)

    def predict(self, X):
        """
        Predict BBB permeability for new compounds.

        Parameters
        ----------
        X : np.ndarray or list
            Features or SMILES strings

        Returns
        -------
        predictions : np.ndarray
            Predicted probabilities of BBB permeability
        """
        # Convert SMILES to features if needed
        if isinstance(X[0], str):
            X = self.featurize_smiles(X)

        # Create dataset and predict
        dataset = dc.data.NumpyDataset(X)
        return self.model.predict(dataset)