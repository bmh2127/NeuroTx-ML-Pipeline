#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
molecular_model_evaluation.py

A comprehensive module for evaluating molecular machine learning models in the
NeuroTx-ML pipeline, with a focus on blood-brain barrier permeability prediction.
This module provides functions for model evaluation, performance visualization,
and interpretation of model predictions.

References:
- DeepChem Metrics: https://deepchem.readthedocs.io/en/latest/api_reference/metrics.html
- DeepChem Model Evaluation: https://deepchem.readthedocs.io/en/latest/api_reference/models.html
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from rdkit import Chem
from rdkit.Chem import Draw
import deepchem as dc
from deepchem.metrics import Metric
from typing import List, Dict, Tuple, Union, Optional, Any


class ModelEvaluator:
    """
    A class for evaluating molecular machine learning models in the NeuroTx-ML pipeline.
    Provides methods for computing various performance metrics, visualizing results,
    and interpreting model predictions.
    """
    
    def __init__(self, model_dir: str = None):
        """
        Initialize the ModelEvaluator.
        
        Parameters
        ----------
        model_dir : str, optional
            Directory where models and evaluation results will be saved
        """
        self.model_dir = model_dir
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    def evaluate_model(self, 
                      model: dc.models.Model, 
                      dataset: dc.data.Dataset, 
                      metrics: List[Metric], 
                      transformers: List[dc.trans.Transformer] = None,
                      per_task_metrics: bool = True) -> Dict[str, Any]:
        """
        Evaluate a model on a dataset using specified metrics.
        
        Parameters
        ----------
        model : dc.models.Model
            The trained model to evaluate
        dataset : dc.data.Dataset
            The dataset to evaluate the model on
        metrics : List[Metric]
            List of metrics to use for evaluation
        transformers : List[dc.trans.Transformer], optional
            List of transformers to apply to the dataset
        per_task_metrics : bool, default=True
            Whether to return metrics for each task separately
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing evaluation results
        """
        results = model.evaluate(dataset, metrics, transformers)
        
        # Format results for better readability
        formatted_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray) and per_task_metrics:
                task_results = {}
                for i, task_value in enumerate(value):
                    task_results[f"task_{i}"] = task_value
                formatted_results[key] = task_results
            else:
                formatted_results[key] = value
        
        return formatted_results
    
    def evaluate_classification_model(self, 
                                     model: dc.models.Model, 
                                     dataset: dc.data.Dataset,
                                     transformers: List[dc.trans.Transformer] = None,
                                     threshold: float = 0.5) -> Dict[str, Any]:
        """
        Evaluate a classification model with standard classification metrics.
        
        Parameters
        ----------
        model : dc.models.Model
            The trained classification model to evaluate
        dataset : dc.data.Dataset
            The dataset to evaluate the model on
        transformers : List[dc.trans.Transformer], optional
            List of transformers to apply to the dataset
        threshold : float, default=0.5
            Classification threshold for binary classification
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing evaluation results
        """
        metrics = [
            Metric(dc.metrics.roc_auc_score, np.mean),
            Metric(dc.metrics.prc_auc_score, np.mean),
            Metric(dc.metrics.accuracy_score, np.mean),
            Metric(dc.metrics.balanced_accuracy_score, np.mean),
            Metric(dc.metrics.precision_score, np.mean),
            Metric(dc.metrics.recall_score, np.mean),
            Metric(dc.metrics.f1_score, np.mean),
            Metric(dc.metrics.matthews_corrcoef, np.mean)
        ]
        
        return self.evaluate_model(model, dataset, metrics, transformers)
    
    def evaluate_regression_model(self, 
                                 model: dc.models.Model, 
                                 dataset: dc.data.Dataset,
                                 transformers: List[dc.trans.Transformer] = None) -> Dict[str, Any]:
        """
        Evaluate a regression model with standard regression metrics.
        
        Parameters
        ----------
        model : dc.models.Model
            The trained regression model to evaluate
        dataset : dc.data.Dataset
            The dataset to evaluate the model on
        transformers : List[dc.trans.Transformer], optional
            List of transformers to apply to the dataset
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing evaluation results
        """
        metrics = [
            Metric(dc.metrics.r2_score, np.mean),
            Metric(dc.metrics.mean_squared_error, np.mean),
            Metric(dc.metrics.mean_absolute_error, np.mean),
            Metric(dc.metrics.pearson_r2_score, np.mean),
            Metric(dc.metrics.rms_score, np.mean)
        ]
        
        return self.evaluate_model(model, dataset, metrics, transformers)
    
    def plot_roc_curve(self, 
                      model: dc.models.Model, 
                      dataset: dc.data.Dataset,
                      task_idx: int = 0,
                      transformers: List[dc.trans.Transformer] = None,
                      title: str = "ROC Curve",
                      save_path: str = None) -> None:
        """
        Plot ROC curve for a binary classification model.
        
        Parameters
        ----------
        model : dc.models.Model
            The trained classification model
        dataset : dc.data.Dataset
            The dataset to evaluate the model on
        task_idx : int, default=0
            Index of the task to plot ROC curve for
        transformers : List[dc.trans.Transformer], optional
            List of transformers to apply to the dataset
        title : str, default="ROC Curve"
            Title of the plot
        save_path : str, optional
            Path to save the plot
        """
        y_true = dataset.y[:, task_idx]
        y_pred = model.predict(dataset, transformers)[:, task_idx]
        
        # For binary classification
        if len(np.unique(y_true)) <= 2:
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                     label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(title)
            plt.legend(loc="lower right")
            
            if save_path:
                plt.savefig(save_path)
            plt.show()
    
    def plot_precision_recall_curve(self, 
                                   model: dc.models.Model, 
                                   dataset: dc.data.Dataset,
                                   task_idx: int = 0,
                                   transformers: List[dc.trans.Transformer] = None,
                                   title: str = "Precision-Recall Curve",
                                   save_path: str = None) -> None:
        """
        Plot precision-recall curve for a binary classification model.
        
        Parameters
        ----------
        model : dc.models.Model
            The trained classification model
        dataset : dc.data.Dataset
            The dataset to evaluate the model on
        task_idx : int, default=0
            Index of the task to plot precision-recall curve for
        transformers : List[dc.trans.Transformer], optional
            List of transformers to apply to the dataset
        title : str, default="Precision-Recall Curve"
            Title of the plot
        save_path : str, optional
            Path to save the plot
        """
        y_true = dataset.y[:, task_idx]
        y_pred = model.predict(dataset, transformers)[:, task_idx]
        
        # For binary classification
        if len(np.unique(y_true)) <= 2:
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            pr_auc = auc(recall, precision)
            
            plt.figure(figsize=(10, 8))
            plt.plot(recall, precision, color='darkorange', lw=2, 
                     label=f'PR curve (area = {pr_auc:.2f})')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(title)
            plt.legend(loc="lower left")
            
            if save_path:
                plt.savefig(save_path)
            plt.show()
    
    def plot_confusion_matrix(self, 
                             model: dc.models.Model, 
                             dataset: dc.data.Dataset,
                             task_idx: int = 0,
                             transformers: List[dc.trans.Transformer] = None,
                             threshold: float = 0.5,
                             title: str = "Confusion Matrix",
                             save_path: str = None) -> None:
        """
        Plot confusion matrix for a binary classification model.
        
        Parameters
        ----------
        model : dc.models.Model
            The trained classification model
        dataset : dc.data.Dataset
            The dataset to evaluate the model on
        task_idx : int, default=0
            Index of the task to plot confusion matrix for
        transformers : List[dc.trans.Transformer], optional
            List of transformers to apply to the dataset
        threshold : float, default=0.5
            Classification threshold for binary classification
        title : str, default="Confusion Matrix"
            Title of the plot
        save_path : str, optional
            Path to save the plot
        """
        y_true = dataset.y[:, task_idx]
        y_pred = model.predict(dataset, transformers)[:, task_idx]
        
        # For binary classification
        if len(np.unique(y_true)) <= 2:
            y_pred_binary = (y_pred > threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred_binary)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(title)
            
            if save_path:
                plt.savefig(save_path)
            plt.show()
    
    def visualize_predictions(self, 
                             model: dc.models.Model, 
                             dataset: dc.data.Dataset,
                             smiles_list: List[str],
                             task_idx: int = 0,
                             transformers: List[dc.trans.Transformer] = None,
                             n_molecules: int = 10,
                             title: str = "Model Predictions",
                             save_path: str = None) -> None:
        """
        Visualize model predictions on molecules.
        
        Parameters
        ----------
        model : dc.models.Model
            The trained model
        dataset : dc.data.Dataset
            The dataset containing the molecules
        smiles_list : List[str]
            List of SMILES strings corresponding to the dataset
        task_idx : int, default=0
            Index of the task to visualize predictions for
        transformers : List[dc.trans.Transformer], optional
            List of transformers to apply to the dataset
        n_molecules : int, default=10
            Number of molecules to visualize
        title : str, default="Model Predictions"
            Title of the plot
        save_path : str, optional
            Path to save the plot
        """
        y_true = dataset.y[:, task_idx]
        y_pred = model.predict(dataset, transformers)[:, task_idx]
        
        # Select a subset of molecules to visualize
        indices = np.random.choice(len(smiles_list), min(n_molecules, len(smiles_list)), replace=False)
        
        mols = [Chem.MolFromSmiles(smiles_list[i]) for i in indices]
        labels = [f"True: {y_true[i]:.2f}\nPred: {y_pred[i]:.2f}" for i in indices]
        
        img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(200, 200), legends=labels)
        
        plt.figure(figsize=(15, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def compare_models(self, 
                      models: List[dc.models.Model], 
                      model_names: List[str],
                      dataset: dc.data.Dataset,
                      metric_name: str,
                      transformers: List[dc.trans.Transformer] = None,
                      title: str = "Model Comparison",
                      save_path: str = None) -> None:
        """
        Compare multiple models using a specific metric.
        
        Parameters
        ----------
        models : List[dc.models.Model]
            List of trained models to compare
        model_names : List[str]
            Names of the models for the plot
        dataset : dc.data.Dataset
            The dataset to evaluate the models on
        metric_name : str
            Name of the metric to use for comparison
        transformers : List[dc.trans.Transformer], optional
            List of transformers to apply to the dataset
        title : str, default="Model Comparison"
            Title of the plot
        save_path : str, optional
            Path to save the plot
        """
        if metric_name == 'roc_auc':
            metric = Metric(dc.metrics.roc_auc_score, np.mean)
        elif metric_name == 'prc_auc':
            metric = Metric(dc.metrics.prc_auc_score, np.mean)
        elif metric_name == 'r2':
            metric = Metric(dc.metrics.r2_score, np.mean)
        elif metric_name == 'mae':
            metric = Metric(dc.metrics.mean_absolute_error, np.mean)
        elif metric_name == 'rmse':
            metric = Metric(dc.metrics.rms_score, np.mean)
        else:
            raise ValueError(f"Unsupported metric: {metric_name}")
        
        scores = []
        for model in models:
            result = model.evaluate(dataset, [metric], transformers)
            scores.append(result[metric_name])
        
        plt.figure(figsize=(10, 6))
        plt.bar(model_names, scores, color='skyblue')
        plt.xlabel('Model')
        plt.ylabel(metric_name)
        plt.title(title)
        plt.xticks(rotation=45)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def cross_validate(self, 
                      model_builder: callable, 
                      dataset: dc.data.Dataset,
                      splitter: dc.splits.Splitter,
                      metrics: List[Metric],
                      transformers: List[dc.trans.Transformer] = None,
                      n_folds: int = 5,
                      seed: int = 123) -> Dict[str, List[float]]:
        """
        Perform cross-validation for a model.
        
        Parameters
        ----------
        model_builder : callable
            Function that builds and returns a new model instance
        dataset : dc.data.Dataset
            The dataset to use for cross-validation
        splitter : dc.splits.Splitter
            Splitter to use for creating folds
        metrics : List[Metric]
            List of metrics to use for evaluation
        transformers : List[dc.trans.Transformer], optional
            List of transformers to apply to the dataset
        n_folds : int, default=5
            Number of folds for cross-validation
        seed : int, default=123
            Random seed for reproducibility
            
        Returns
        -------
        Dict[str, List[float]]
            Dictionary containing evaluation results for each fold
        """
        np.random.seed(seed)
        
        # Generate folds
        fold_datasets = splitter.k_fold_split(dataset, k=n_folds)
        
        all_results = {}
        for metric in metrics:
            all_results[metric.name] = []
        
        for fold_num, (train_dataset, valid_dataset) in enumerate(fold_datasets):
            print(f"Fold {fold_num + 1}/{n_folds}")
            
            # Build a new model for each fold
            model = model_builder()
            
            # Train the model
            model.fit(train_dataset)
            
            # Evaluate the model
            results = model.evaluate(valid_dataset, metrics, transformers)
            
            # Store results
            for metric in metrics:
                all_results[metric.name].append(results[metric.name])
        
        # Calculate mean and std for each metric
        summary = {}
        for metric_name, values in all_results.items():
            summary[f"{metric_name}_mean"] = np.mean(values)
            summary[f"{metric_name}_std"] = np.std(values)
        
        return {"fold_results": all_results, "summary": summary}


def load_bbbp_model_for_evaluation(model_path: str) -> dc.models.Model:
    """
    Load a pre-trained BBB permeability model for evaluation.
    
    Parameters
    ----------
    model_path : str
        Path to the saved model
        
    Returns
    -------
    dc.models.Model
        Loaded model
    """
    # Load the model
    model = dc.models.GraphConvModel(n_tasks=1, mode='classification')
    model.restore(model_path)
    return model


def evaluate_bbbp_model(model: dc.models.Model, 
                       test_dataset: dc.data.Dataset,
                       transformers: List[dc.trans.Transformer] = None,
                       output_dir: str = None) -> Dict[str, Any]:
    """
    Evaluate a BBB permeability model on a test dataset.
    
    Parameters
    ----------
    model : dc.models.Model
        The trained BBB permeability model
    test_dataset : dc.data.Dataset
        The test dataset
    transformers : List[dc.trans.Transformer], optional
        List of transformers to apply to the dataset
    output_dir : str, optional
        Directory to save evaluation results
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing evaluation results
    """
    evaluator = ModelEvaluator(model_dir=output_dir)
    
    # Evaluate the model
    results = evaluator.evaluate_classification_model(model, test_dataset, transformers)
    
    # Plot ROC curve
    if output_dir:
        evaluator.plot_roc_curve(
            model, test_dataset, transformers=transformers,
            title="BBB Permeability ROC Curve",
            save_path=os.path.join(output_dir, "bbbp_roc_curve.png")
        )
        
        # Plot precision-recall curve
        evaluator.plot_precision_recall_curve(
            model, test_dataset, transformers=transformers,
            title="BBB Permeability Precision-Recall Curve",
            save_path=os.path.join(output_dir, "bbbp_pr_curve.png")
        )
        
        # Plot confusion matrix
        evaluator.plot_confusion_matrix(
            model, test_dataset, transformers=transformers,
            title="BBB Permeability Confusion Matrix",
            save_path=os.path.join(output_dir, "bbbp_confusion_matrix.png")
        )
    
    return results


def main():
    """
    Main function to demonstrate model evaluation.
    """
    # Load BBBP dataset
    tasks, datasets, transformers = dc.molnet.load_bbbp(featurizer='GraphConv', split='scaffold')
    train_dataset, valid_dataset, test_dataset = datasets
    
    # Create and train a model
    model = dc.models.GraphConvModel(n_tasks=1, batch_size=64, mode='classification')
    model.fit(train_dataset, nb_epoch=10)
    
    # Evaluate the model
    results = evaluate_bbbp_model(model, test_dataset, transformers, output_dir="./evaluation_results")
    
    print("Evaluation Results:")
    for metric_name, value in results.items():
        print(f"{metric_name}: {value}")


if __name__ == "__main__":
    main()