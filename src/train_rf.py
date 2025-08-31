import os
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem import DataStructs
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed

MAX_MZ = 2000
ADDUCTS = ['[M+H]+', '[M+Na]+', 'M+H', 'M-H', '[M-H2O+H]+', '[M-H]-', '[M+NH4]+', 'M+NH4', 'M+Na']
FRAGMENT_LEVELS = [-4, -3, -2, -1, 0]
FINGERPRINT_NBITS = 1024
RANDOM_SEED = 43242

class SKLearnRFSpec2Mol:
    """Random Forest model using scikit-learn for predicting molecular fingerprints from mass spectra"""
    
    def __init__(self, rf_params=None, model_dir="models/sklearn_rf"):
        """
        Initialize the Random Forest model
        
        Args:
            rf_params: Dictionary of Random Forest parameters
            model_dir: Directory to save models
        """
        self.rf_params = rf_params or {
            'n_estimators': 100,
            'max_depth': 16,
            'max_features': 'sqrt',
            'min_samples_leaf': 3,
            'min_samples_split': 5,
            'random_state': RANDOM_SEED,
            'n_jobs': -1  # Use all available cores
        }
        self.model_dir = model_dir
        self.models = []  # Will contain one RF model for each fingerprint bit
        os.makedirs(model_dir, exist_ok=True)
    
    def load_data_from_tsv(self, tsv_file, encode_spec_func, mol_representation_func):
        """
        Load training data from TSV file
        
        Args:
            tsv_file: Path to TSV file
            encode_spec_func: Function to encode spectra 
            mol_representation_func: Function to compute molecular representation
        
        Returns:
            X: Spectrum features
            y: Molecular fingerprint targets
        """
        print(f"Loading data from {tsv_file}...")
        
        spectra = []
        fingerprints = []
        
        with open(tsv_file, 'r') as f:
            for line in tqdm(f.readlines()):
                try:
                    parts = line.strip().split('\t')
                    if len(parts) != 4:
                        continue
                    
                    sid, adduct, spec_str, smiles = parts
                    
                    # Parse spectrum
                    spec = np.array(json.loads(spec_str))
                    if not (len(spec.shape) == 2 and spec.shape[0] >= 1 and (spec <= 0).sum() == 0):
                        continue
                    
                    # Round to 3 digits M/Z precision and normalize intensities
                    spec[:, 0] = np.round(spec[:, 0], 3)
                    spec[:, 1] /= spec[:, 1].max()
                    
                    # Convert to RDKit mol object
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        continue
                    
                    # Encode spectrum
                    encoded_spec = encode_spec_func(spec)
                    spectra.append(encoded_spec)
                    
                    # Compute fragmentation levels for fingerprint
                    frag_levels = self._get_featurized_fragmentation_level_from_mol(mol, encoded_spec)
                    adduct_feats = self._get_featurized_adducts(adduct)
                    
                    # Compute molecular fingerprint
                    fp = mol_representation_func(mol, frag_levels=frag_levels, adduct_feats=adduct_feats)
                    fingerprints.append(fp)
                    
                except Exception as e:
                    print(f"Error processing line: {e}")
                    continue
        
        return np.array(spectra), np.array(fingerprints)
    
    def _get_fragmentation_level(self, mol, spec):
        """Get fragmentation level for a molecule and spectrum"""
        mass = Descriptors.ExactMolWt(mol)
        min_mass, max_mass = int(max(0, mass - 25)), int(min(mass + 25, MAX_MZ))
        _spec = 10 ** spec - 1
        frag_level = max(0.01, _spec[min_mass:max_mass + 1].sum())
        return np.log10(frag_level / _spec.sum())
    
    def _get_featurized_fragmentation_level(self, frag_level):
        """Convert fragmentation level to one-hot encoding"""
        flf = [int(frag_level <= FRAGMENT_LEVELS[0])]
        flf += [int(FRAGMENT_LEVELS[i-1] <= frag_level <= FRAGMENT_LEVELS[i]) 
                for i in range(1, len(FRAGMENT_LEVELS))]
        return np.array(flf)
    
    def _get_featurized_adducts(self, adduct):
        """Convert adduct to one-hot encoding"""
        return np.array([int(adduct == ADDUCTS[i]) for i in range(len(ADDUCTS))])
    
    def _get_featurized_fragmentation_level_from_mol(self, mol, spec):
        """Get fragmentation level features for a molecule"""
        fl = self._get_fragmentation_level(mol, spec)
        return self._get_featurized_fragmentation_level(fl)
    
    def _train_single_model(self, X_train, y_train_bit, X_val, y_val_bit, bit_idx):
        """Train single RF model for one fingerprint bit"""
        # Train model
        rf = RandomForestRegressor(**self.rf_params)
        rf.fit(X_train, y_train_bit)
        
        # Save model
        model_path = os.path.join(self.model_dir, f"rf_bit_{bit_idx}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(rf, f)
        
        # Calculate MSE
        y_pred = rf.predict(X_val)
        mse = mean_squared_error(y_val_bit, y_pred)
        
        return rf, mse, bit_idx
    
    def train(self, X, y, validation_split=0.2, n_bits_batch=None, n_jobs=4):
        """
        Train the Random Forest model
        
        Args:
            X: Input features (spectra)
            y: Target values (fingerprints)
            validation_split: Fraction of data to use for validation
            n_bits_batch: Number of fingerprint bits to train (None = train all bits)
            n_jobs: Number of parallel jobs for training multiple models
        """
        print(f"Training Random Forest models on {X.shape[0]} samples...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=RANDOM_SEED
        )
        
        # Calculate dimensionality of fingerprints
        fp_dim = y_train.shape[1]
        print(f"Fingerprint dimension: {fp_dim}")
        
        # Determine how many bits to train
        if n_bits_batch is None or n_bits_batch >= fp_dim:
            indices = np.arange(fp_dim)
            print(f"Training all {fp_dim} fingerprint bits")
        else:
            # Train separate models for a subset of fingerprint bits
            indices = np.linspace(0, fp_dim-1, n_bits_batch, dtype=int)
            print(f"Training {len(indices)} out of {fp_dim} fingerprint bits")
        
        # Train models in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._train_single_model)(
                X_train, y_train[:, i], X_val, y_val[:, i], i
            ) for i in indices
        )
        
        # Unpack results
        self.models = []
        mse_scores = []
        trained_indices = []
        
        for rf, mse, idx in results:
            self.models.append(rf)
            mse_scores.append(mse)
            trained_indices.append(idx)
        
        # Calculate average MSE
        avg_mse = np.mean(mse_scores)
        print(f"Average MSE across fingerprint bits: {avg_mse:.6f}")
        
        # Save model metadata
        metadata = {
            'avg_mse': avg_mse,
            'trained_indices': [int(idx) for idx in trained_indices],
            'params': {k: (float(v) if isinstance(v, (np.number, float)) else 
                      (int(v) if isinstance(v, (np.integer, int)) else v)) 
                  for k, v in self.rf_params.items() if not callable(v)},
        }
        
        with open(os.path.join(self.model_dir, 'rf_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return avg_mse
    
    def predict(self, X):
        """
        Predict molecular fingerprints from spectra
        
        Args:
            X: Input features (spectra)
            
        Returns:
            y_pred: Predicted fingerprints
        """
        if not self.models:
            raise ValueError("Models not trained yet!")
        
        # Initialize output array
        fp_dim = 1024 + 857 + 167 + len(FRAGMENT_LEVELS) + len(ADDUCTS)
        y_pred = np.zeros((X.shape[0], fp_dim))
        
        # Load metadata to get indices
        with open(os.path.join(self.model_dir, 'rf_metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        # Predict in parallel using joblib
        def predict_single_bit(model, X, idx):
            preds = model.predict(X)
            return idx, preds
        
        results = Parallel(n_jobs=-1)(
            delayed(predict_single_bit)(model, X, metadata['trained_indices'][i])
            for i, model in enumerate(self.models)
        )
        
        # Combine predictions
        for idx, preds in results:
            y_pred[:, idx] = preds
        
        return y_pred
        
    def load_models(self, model_dir=None):
        """Load trained models from directory"""
        model_dir = model_dir or self.model_dir
        
        # Load metadata
        with open(os.path.join(model_dir, 'rf_metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        # Load models
        self.models = []
        for idx in metadata['trained_indices']:
            model_path = os.path.join(model_dir, f"rf_bit_{idx}.pkl")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            self.models.append(model)
            
        return metadata

    def get_feature_importance(self):
        """Get feature importance from trained models"""
        if not self.models:
            raise ValueError("Models not trained yet!")
            
        # Get feature importance from all models
        importances = []
        
        # Load metadata to get indices
        with open(os.path.join(self.model_dir, 'rf_metadata.json'), 'r') as f:
            metadata = json.load(f)
            
        for i, model in enumerate(self.models):
            idx = metadata['trained_indices'][i]
            importance = model.feature_importances_
            importances.append((idx, importance))
            
        return importances


def hunhe_fingerprint1(mol, frag_levels, adduct_feats):
    """Combined fingerprint representation (copied from your code)"""
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    torsion_fp = AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=857)
    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    
    mol_rep = np.zeros((0,))
    torsion_rep = np.zeros((0,))
    mass_rep = np.zeros((0,))
    
    DataStructs.ConvertToNumpyArray(morgan_fp, mol_rep)
    DataStructs.ConvertToNumpyArray(torsion_fp, torsion_rep)
    DataStructs.ConvertToNumpyArray(maccs_fp, mass_rep)
    
    return np.hstack([mol_rep, torsion_rep, mass_rep, frag_levels, adduct_feats])


def encode_spec(spec):
    """Encode spectrum (copied from your code)"""
    vec = np.zeros(MAX_MZ * 2)
    for i in range(spec.shape[0]):
        mz_rnd = int(spec[i, 0])
        if mz_rnd >= MAX_MZ:
            continue
        logint = np.log10(spec[i, 1] + 1)
        if vec[mz_rnd] < logint:
            vec[mz_rnd] = logint
            vec[MAX_MZ + mz_rnd] = np.log10((spec[i, 0] - mz_rnd) + 1)
    return vec


def plot_feature_importance(model_dir, top_n=20):
    """Plot feature importance from Random Forest models"""
    # Create RF model to load trained models
    model = SKLearnRFSpec2Mol(model_dir=model_dir)
    model.load_models()
    
    # Get feature importance
    importances = model.get_feature_importance()
    
    # Plot feature importance for a few models
    plt.figure(figsize=(12, 8))
    for idx, importance in importances[:5]:  # Plot first 5 models
        # Get top N important features
        top_indices = np.argsort(importance)[-top_n:]
        top_importances = importance[top_indices]
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.barh(range(len(top_indices)), top_importances, align='center')
        plt.yticks(range(len(top_indices)), top_indices)
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature Index')
        plt.title(f'Top {top_n} Important Features for Bit {idx}')
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, f'feature_importance_bit_{idx}.png'))
        plt.close()
    
    print(f"Feature importance plots saved to {model_dir}")


def main():
    """Main function to train and evaluate Random Forest model"""
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='/data/zhangxiaofeng/code/code/data/train_data/merged_train.tsv',
                      help='Path to training data TSV file')
    parser.add_argument('--model_dir', type=str, default='models/sklearn_rf',
                      help='Directory to save models')
    parser.add_argument('--max_samples', type=int, default=None,
                      help='Maximum number of samples to use for training')
    parser.add_argument('--n_bits_batch', type=int, default=None,
                      help='Number of fingerprint bits to train (None = all bits)')
    parser.add_argument('--n_estimators', type=int, default=100,
                      help='Number of trees in Random Forest')
    parser.add_argument('--max_depth', type=int, default=16,
                      help='Maximum depth of trees')
    parser.add_argument('--val_split', type=float, default=0.2,
                      help='Validation split ratio')
    parser.add_argument('--n_jobs', type=int, default=4,
                      help='Number of parallel jobs for training')
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'evaluate'], 
                      default='train', help='Mode of operation')
    parser.add_argument('--test_file', type=str, default=None,
                      help='Path to test data for evaluation')
    args = parser.parse_args()
    
    # Create model with custom parameters
    rf_params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'max_features': 'sqrt',
        'min_samples_leaf': 3,
        'min_samples_split': 5,
        'random_state': RANDOM_SEED,
        'n_jobs': 1,  # Set to 1 because we parallelize at a higher level
    }
    
    model = SKLearnRFSpec2Mol(rf_params=rf_params, model_dir=args.model_dir)
    
    if args.mode == 'train':
        # Load data from TSV
        X, y = model.load_data_from_tsv(
            args.train_file, 
            encode_spec_func=encode_spec, 
            mol_representation_func=hunhe_fingerprint1
        )
        
        # Limit number of samples if specified
        if args.max_samples and args.max_samples < X.shape[0]:
            indices = np.random.choice(X.shape[0], args.max_samples, replace=False)
            X = X[indices]
            y = y[indices]
        
        # Train model
        model.train(
            X, y, 
            validation_split=args.val_split,
            n_bits_batch=args.n_bits_batch,
            n_jobs=args.n_jobs
        )
        
        # Plot feature importance
        plot_feature_importance(args.model_dir)
        print(f"Training complete. Models saved to {args.model_dir}")
        
    elif args.mode == 'predict':
        if not args.test_file:
            raise ValueError("Test file must be specified for predict mode")
            
        # Load models
        model.load_models()
        
        # Load test data
        X_test, y_test = model.load_data_from_tsv(
            args.test_file,
            encode_spec_func=encode_spec,
            mol_representation_func=hunhe_fingerprint1
        )
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Save predictions
        np.save(os.path.join(args.model_dir, 'predictions.npy'), y_pred)
        np.save(os.path.join(args.model_dir, 'true_values.npy'), y_test)
        print(f"Predictions saved to {os.path.join(args.model_dir, 'predictions.npy')}")
        
    elif args.mode == 'evaluate':
        if not args.test_file:
            raise ValueError("Test file must be specified for evaluate mode")
            
        # Load models
        model.load_models()
        
        # Load test data
        X_test, y_test = model.load_data_from_tsv(
            args.test_file,
            encode_spec_func=encode_spec,
            mol_representation_func=hunhe_fingerprint1
        )
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Calculate cosine similarity
        cosine_similarities = []
        for i in range(y_test.shape[0]):
            similarity = np.dot(y_test[i], y_pred[i]) / (np.linalg.norm(y_test[i]) * np.linalg.norm(y_pred[i]))
            cosine_similarities.append(similarity)
        
        avg_cosine = np.mean(cosine_similarities)
        
        print(f"Evaluation results:")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  Average Cosine Similarity: {avg_cosine:.6f}")
        
        # Save evaluation results
        results = {
            'mse': mse,
            'mae': mae,
            'avg_cosine_similarity': avg_cosine,
            'cosine_similarities': cosine_similarities,
        }
        
        with open(os.path.join(args.model_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Evaluation results saved to {os.path.join(args.model_dir, 'evaluation_results.json')}")


if __name__ == '__main__':
    main()