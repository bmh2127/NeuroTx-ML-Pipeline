    # data/templates/bbb_dataset_processing.py

import os
import numpy as np
import pandas as pd
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from typing import Tuple, List, Dict, Optional, Union

class BBBDatasetProcessor:
    """
    Processor for blood-brain barrier permeability datasets.
    
    This class handles loading, preprocessing, and featurizing compounds
    for blood-brain barrier (BBB) permeability prediction. It leverages
    DeepChem's MoleculeNet BBBP dataset and provides additional functionality
    for custom datasets.
    
    Attributes
    ----------
    data_dir : str
        Directory to store dataset files
    featurizer : dc.feat.Featurizer
        Featurizer to use for molecular representation
    splitter : dc.splits.Splitter
        Splitter to use for train/valid/test splitting
    transformers : List[dc.trans.Transformer]
        List of transformers to apply to the dataset
    """
    
    def __init__(
        self,
        data_dir: str = 'data',
        featurizer_type: str = 'GraphConv',
        splitter_type: str = 'scaffold',
        transformers: List[str] = ['balancing'],
        reload: bool = True
    ):
        """
        Initialize the BBB dataset processor.
        
        Parameters
        ----------
        data_dir : str
            Directory to store dataset files
        featurizer_type : str
            Type of featurizer to use ('ECFP', 'GraphConv', 'Weave', etc.)
        splitter_type : str
            Type of splitter to use ('scaffold', 'random', 'stratified', etc.)
        transformers : List[str]
            List of transformers to apply to the dataset
        reload : bool
            Whether to reload cached datasets
        """
        self.data_dir = data_dir
        self.featurizer_type = featurizer_type
        self.splitter_type = splitter_type
        self.transformer_types = transformers
        self.reload = reload
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize featurizer
        self.featurizer = self._get_featurizer(featurizer_type)
        
        # Initialize splitter
        self.splitter = self._get_splitter(splitter_type)
        
    def _get_featurizer(self, featurizer_type: str) -> dc.feat.Featurizer:
        """
        Get the appropriate featurizer based on the specified type.
        
        Parameters
        ----------
        featurizer_type : str
            Type of featurizer to use
            
        Returns
        -------
        dc.feat.Featurizer
            Initialized featurizer
        """
        if featurizer_type == 'ECFP':
            return dc.feat.CircularFingerprint(size=1024, radius=2)
        elif featurizer_type == 'GraphConv':
            return dc.feat.MolGraphConvFeaturizer()
        elif featurizer_type == 'Weave':
            return dc.feat.WeaveFeaturizer()
        elif featurizer_type == 'RDKit':
            return dc.feat.RDKitDescriptors()
        else:
            raise ValueError(f"Unsupported featurizer type: {featurizer_type}")
    
    def _get_splitter(self, splitter_type: str) -> dc.splits.Splitter:
        """
        Get the appropriate splitter based on the specified type.
        
        Parameters
        ----------
        splitter_type : str
            Type of splitter to use
            
        Returns
        -------
        dc.splits.Splitter
            Initialized splitter
        """
        if splitter_type == 'scaffold':
            return dc.splits.ScaffoldSplitter()
        elif splitter_type == 'random':
            return dc.splits.RandomSplitter()
        elif splitter_type == 'stratified':
            return dc.splits.RandomStratifiedSplitter()
        else:
            raise ValueError(f"Unsupported splitter type: {splitter_type}")
    
    def load_moleculenet_bbbp(self) -> Tuple[List[str], Tuple[dc.data.Dataset, dc.data.Dataset, dc.data.Dataset], List[dc.trans.Transformer]]:
        """
        Load the MoleculeNet BBBP dataset.
        
        Returns
        -------
        Tuple[List[str], Tuple[dc.data.Dataset, dc.data.Dataset, dc.data.Dataset], List[dc.trans.Transformer]]
            A tuple containing:
            - List of task names
            - Tuple of (train, valid, test) datasets
            - List of transformers applied to the datasets
        """
        print("Loading MoleculeNet BBBP dataset...")
        tasks, datasets, transformers = dc.molnet.load_bbbp(
            featurizer=self.featurizer,
            splitter=self.splitter,
            transformers=self.transformer_types,
            reload=self.reload,
            data_dir=self.data_dir
        )
        
        train_dataset, valid_dataset, test_dataset = datasets
        
        print(f"Dataset loaded: {len(train_dataset)} train, {len(valid_dataset)} valid, {len(test_dataset)} test samples")
        
        return tasks, datasets, transformers
    
    def load_custom_bbbp(self, csv_file: str, smiles_col: str = 'smiles', label_col: str = 'p_np') -> Tuple[List[str], Tuple[dc.data.Dataset, dc.data.Dataset, dc.data.Dataset], List[dc.trans.Transformer]]:
        """
        Load a custom BBB permeability dataset from a CSV file.
        
        Parameters
        ----------
        csv_file : str
            Path to the CSV file containing the dataset
        smiles_col : str
            Name of the column containing SMILES strings
        label_col : str
            Name of the column containing permeability labels
            
        Returns
        -------
        Tuple[List[str], Tuple[dc.data.Dataset, dc.data.Dataset, dc.data.Dataset], List[dc.trans.Transformer]]
            A tuple containing:
            - List of task names
            - Tuple of (train, valid, test) datasets
            - List of transformers applied to the datasets
        """
        print(f"Loading custom BBBP dataset from {csv_file}...")
        
        # Define tasks
        tasks = [label_col]
        
        # Load data
        loader = dc.data.CSVLoader(
            tasks=tasks,
            feature_field=smiles_col,
            featurizer=self.featurizer
        )
        dataset = loader.create_dataset(csv_file, shard_size=8192)
        
        # Initialize transformers
        transformers = []
        for transformer_type in self.transformer_types:
            if transformer_type == 'balancing':
                transformer = dc.trans.BalancingTransformer(dataset=dataset)
                dataset = transformer.transform(dataset)
                transformers.append(transformer)
        
        # Split dataset
        train_dataset, valid_dataset, test_dataset = self.splitter.train_valid_test_split(
            dataset, 
            frac_train=0.8,
            frac_valid=0.1,
            frac_test=0.1
        )
        
        print(f"Dataset loaded: {len(train_dataset)} train, {len(valid_dataset)} valid, {len(test_dataset)} test samples")
        
        return tasks, (train_dataset, valid_dataset, test_dataset), transformers
    
    def analyze_dataset(self, dataset: dc.data.Dataset) -> Dict:
        """
        Analyze a dataset to extract useful statistics and properties.
        
        Parameters
        ----------
        dataset : dc.data.Dataset
            Dataset to analyze
            
        Returns
        -------
        Dict
            Dictionary containing dataset statistics and properties
        """
        stats = {}
        
        # Basic statistics
        stats['num_samples'] = len(dataset)
        
        if hasattr(dataset, 'y') and dataset.y is not None:
            y = dataset.y
            stats['positive_samples'] = int(np.sum(y == 1))
            stats['negative_samples'] = int(np.sum(y == 0))
            stats['positive_ratio'] = float(stats['positive_samples'] / stats['num_samples'])
        
        return stats
    
    def calculate_molecular_properties(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        Calculate molecular properties for a list of SMILES strings.
        
        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES strings
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing calculated molecular properties
        """
        properties = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                properties.append({
                    'smiles': smiles,
                    'valid': False
                })
                continue
                
            prop_dict = {
                'smiles': smiles,
                'valid': True,
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'h_bond_donors': Lipinski.NumHDonors(mol),
                'h_bond_acceptors': Lipinski.NumHAcceptors(mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'aromatic_rings': Chem.Lipinski.NumAromaticRings(mol),
                'heavy_atoms': mol.GetNumHeavyAtoms(),
                'molar_refractivity': Descriptors.MolMR(mol),
                'topological_polar_surface_area': Descriptors.TPSA(mol)
            }
            
            properties.append(prop_dict)
        
        return pd.DataFrame(properties)
    
    def visualize_compounds(self, smiles_list: List[str], labels: Optional[List[int]] = None, n_per_row: int = 4, molsPerGrid: int = 16):
        """
        Visualize compounds as a grid of molecular structures.
        
        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES strings to visualize
        labels : Optional[List[int]]
            Optional list of labels (0 or 1) for each compound
        n_per_row : int
            Number of molecules per row in the grid
        molsPerGrid : int
            Maximum number of molecules per grid
        """
        try:
            from rdkit.Chem import Draw
            import matplotlib.pyplot as plt
            
            mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list[:molsPerGrid]]
            valid_mols = [mol for mol in mols if mol is not None]
            
            if labels is not None:
                # Truncate labels to match valid molecules
                valid_labels = [labels[i] for i, mol in enumerate(mols) if mol is not None]
                legends = [f"BBB+: {label==1}" for label in valid_labels]
            else:
                legends = ['' for _ in valid_mols]
            
            img = Draw.MolsToGridImage(
                valid_mols,
                molsPerRow=n_per_row,
                subImgSize=(200, 200),
                legends=legends
            )
            
            plt.figure(figsize=(15, 15))
            plt.imshow(img)
            plt.axis('off')
            plt.title("BBB Permeability Dataset Compounds")
            plt.show()
            
        except ImportError:
            print("RDKit Draw or matplotlib not available. Cannot visualize compounds.")
    
    def get_lipinski_rule_of_five(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        Check Lipinski's Rule of Five for a list of SMILES strings.
        
        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES strings
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing Lipinski Rule of Five violations
        """
        results = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
                
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            h_donors = Lipinski.NumHDonors(mol)
            h_acceptors = Lipinski.NumHAcceptors(mol)
            
            violations = 0
            if mw > 500: violations += 1
            if logp > 5: violations += 1
            if h_donors > 5: violations += 1
            if h_acceptors > 10: violations += 1
            
            results.append({
                'smiles': smiles,
                'molecular_weight': mw,
                'logp': logp,
                'h_bond_donors': h_donors,
                'h_bond_acceptors': h_acceptors,
                'violations': violations,
                'passes_rule_of_five': violations <= 1
            })
        
        return pd.DataFrame(results)


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = BBBDatasetProcessor(
        data_dir='data',
        featurizer_type='GraphConv',
        splitter_type='scaffold'
    )
    
    # Load MoleculeNet BBBP dataset
    tasks, datasets, transformers = processor.load_moleculenet_bbbp()
    train_dataset, valid_dataset, test_dataset = datasets
    
    # Analyze dataset
    train_stats = processor.analyze_dataset(train_dataset)
    print(f"Training set statistics: {train_stats}")
    
    # Extract SMILES from the original dataset
    import pandas as pd
    bbbp_df = pd.read_csv(os.path.join(processor.data_dir, 'BBBP.csv'))
    smiles_list = bbbp_df['smiles'].tolist()[:10]  # Just take first 10 for example
    labels = bbbp_df['p_np'].tolist()[:10]
    
    # Calculate molecular properties
    properties_df = processor.calculate_molecular_properties(smiles_list)
    print("\nMolecular properties:")
    print(properties_df.head())
    
    # Check Lipinski's Rule of Five
    lipinski_df = processor.get_lipinski_rule_of_five(smiles_list)
    print("\nLipinski's Rule of Five:")
    print(lipinski_df.head())
    
    # Visualize compounds
    processor.visualize_compounds(smiles_list, labels)