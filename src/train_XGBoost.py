import os
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem import DataStructs
from rdkit.Chem import Descriptors

MAX_MZ = 2000
ADDUCTS = ['[M+H]+', '[M+Na]+', 'M+H', 'M-H', '[M-H2O+H]+', '[M-H]-', '[M+NH4]+', 'M+NH4', 'M+Na']
FRAGMENT_LEVELS = [-4, -3, -2, -1, 0]
FINGERPRINT_NBITS = 1024
RANDOM_SEED = 43242

class XGBSpec2Mol:
    """XGBoost model for predicting molecular fingerprints from mass spectra"""
    
    def __init__(self, xgb_params=None, model_dir="models"):
        """
        Initialize the XGBoost model
        
        Args:
            xgb_params: Dictionary of XGBoost parameters
            model_dir: Directory to save models
        """
        self.xgb_params = xgb_params or {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'eta': 0.1,
            'max_depth': 8,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'gpu_hist',  # Use 'gpu_hist' if GPU available
            'seed': RANDOM_SEED,
        }
        self.model_dir = model_dir
        self.models = []  # Will contain one XGBoost model for each fingerprint bit
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
    
    def train(self, X, y, validation_split=0.2, early_stopping_rounds=10, n_estimators=500):
        """
        Train the XGBoost model
        
        Args:
            X: Input features (spectra)
            y: Target values (fingerprints)
            validation_split: Fraction of data to use for validation
            early_stopping_rounds: Number of rounds for early stopping
            n_estimators: Maximum number of boosting rounds
        """
        print(f"Training XGBoost models on {X.shape[0]} samples...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=RANDOM_SEED
        )
        
        # Calculate dimensionality of fingerprints
        fp_dim = y_train.shape[1]
        print(f"Fingerprint dimension: {fp_dim}")
        
        # Train separate models for each bit of the fingerprint
        # For large fingerprints, consider training models in batches or for selected bits only
        train_batch_size = min(100, fp_dim)  # To avoid training 2000+ models
        indices = np.linspace(0, fp_dim-1, train_batch_size, dtype=int)
        
        self.models = []
        mse_scores = []
        
        for i in tqdm(indices, desc="Training XGBoost models"):
            # Create DMatrix for faster training
            dtrain = xgb.DMatrix(X_train, y_train[:, i])
            dval = xgb.DMatrix(X_val, y_val[:, i])
            
            # Train model
            model = xgb.train(
                self.xgb_params,
                dtrain,
                num_boost_round=n_estimators,
                evals=[(dval, 'val')],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False
            )
            
            # Save model
            model_path = os.path.join(self.model_dir, f"xgb_bit_{i}.json")
            model.save_model(model_path)
            
            # Calculate MSE
            y_pred = model.predict(dval)
            mse = mean_squared_error(y_val[:, i], y_pred)
            mse_scores.append(mse)
            
            self.models.append(model)
        
        # Calculate average MSE
        avg_mse = np.mean(mse_scores)
        print(f"Average MSE across fingerprint bits: {avg_mse:.6f}")
        
        # Save model metadata
        metadata = {
            'avg_mse': avg_mse,
            'trained_indices': indices.tolist(),
            'params': self.xgb_params,
        }
        
        with open(os.path.join(self.model_dir, 'xgb_metadata.json'), 'w') as f:
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
        
        # 首先从模型文件名获取索引
        for i, model in enumerate(self.models):
            # 获取训练时保存的这个模型对应的特征索引
            # 尝试从模型文件名获取索引
            try:
                # 如果模型已经保存为文件，则从文件名获取
                if hasattr(model, 'save_config'):
                    config = model.save_config()
                    # 检查config是否为字符串 
                    if isinstance(config, str):
                        import json
                        config_dict = json.loads(config)
                        if 'learner' in config_dict and 'feature_names' in config_dict['learner']:
                            feature_name = config_dict['learner']['feature_names'][0]
                            # 从特征名称中提取索引
                            idx = int(os.path.basename(feature_name).split('_')[1])
                        else:
                            # 如果没有找到特征名称，则使用元数据中存储的索引
                            with open(os.path.join(self.model_dir, 'xgb_metadata.json'), 'r') as f:
                                metadata = json.load(f)
                                idx = metadata['trained_indices'][i]
                    else:
                        # 配置是字典
                        if 'learner' in config and 'feature_names' in config['learner']:
                            feature_name = config['learner']['feature_names'][0]
                            idx = int(os.path.basename(feature_name).split('_')[1])
                        else:
                            # 使用元数据中的索引
                            with open(os.path.join(self.model_dir, 'xgb_metadata.json'), 'r') as f:
                                metadata = json.load(f)
                                idx = metadata['trained_indices'][i]
                else:
                    # 使用元数据中的索引
                    with open(os.path.join(self.model_dir, 'xgb_metadata.json'), 'r') as f:
                        metadata = json.load(f)
                        idx = metadata['trained_indices'][i]
            except Exception as e:
                # 如果无法从配置或文件名获取，使用训练时元数据中存储的索引
                print(f"Error getting index from model: {e}")
                with open(os.path.join(self.model_dir, 'xgb_metadata.json'), 'r') as f:
                    metadata = json.load(f)
                    idx = metadata['trained_indices'][i]
            
            # 使用模型进行预测
            dtest = xgb.DMatrix(X)
            y_pred[:, idx] = model.predict(dtest)
        
        return y_pred
        
    def load_models(self, model_dir=None):
        """Load trained models from directory"""
        model_dir = model_dir or self.model_dir
        
        # Load metadata
        with open(os.path.join(model_dir, 'xgb_metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        # Load models
        self.models = []
        for idx in metadata['trained_indices']:
            model_path = os.path.join(model_dir, f"xgb_bit_{idx}.json")
            model = xgb.Booster()
            model.load_model(model_path)
            self.models.append(model)
            
        return metadata


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


def plot_learning_curves(model_dir):
    """Plot learning curves from XGBoost training logs"""
    # Load metadata
    with open(os.path.join(model_dir, 'xgb_metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    # Plot learning curves for a few models
    plt.figure(figsize=(12, 8))
    for i in range(min(5, len(metadata['trained_indices']))):
        idx = metadata['trained_indices'][i]
        model_path = os.path.join(model_dir, f"xgb_bit_{idx}.json")
        model = xgb.Booster()
        model.load_model(model_path)
        
        # Get evaluation results
        results = model.get_score(importance_type='gain')
        plt.bar(range(len(results)), list(results.values()), align='center')
        plt.xticks(range(len(results)), list(results.keys()), rotation=90)
        
        plt.title(f'Feature Importance for Bit {idx}')
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, f'feature_importance_bit_{idx}.png'))
        plt.close()
    
    print(f"Feature importance plots saved to {model_dir}")


def main():
    """Main function to train and evaluate XGBoost model"""
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='/data/zhangxiaofeng/code/code/data/train_data/merged_train.tsv',
                      help='Path to training data TSV file')
    parser.add_argument('--model_dir', type=str, default='models/xgboost',
                      help='Directory to save models')
    parser.add_argument('--max_samples', type=int, default=None,
                      help='Maximum number of samples to use for training')
    parser.add_argument('--n_estimators', type=int, default=500,
                      help='Maximum number of trees in XGBoost')
    parser.add_argument('--early_stopping', type=int, default=10,
                      help='Early stopping rounds')
    parser.add_argument('--val_split', type=float, default=0.2,
                      help='Validation split ratio')
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'evaluate'], 
                      default='train', help='Mode of operation')
    parser.add_argument('--test_file', type=str, default=None,
                      help='Path to test data for evaluation')
    args = parser.parse_args()
    
    # Create model
    model = XGBSpec2Mol(model_dir=args.model_dir)
    
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
            early_stopping_rounds=args.early_stopping,
            n_estimators=args.n_estimators
        )
        
        # Plot learning curves
        plot_learning_curves(args.model_dir)
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


# python xgb_spec2mol.py --mode train --train_file /path/to/training/data.tsv --model_dir models/xgboost
# python xgb_spec2mol.py --mode predict --test_file /path/to/test/data.tsv --model_dir models/xgboost
# python xgb_spec2mol.py --mode evaluate --test_file /path/to/test/data.tsv --model_dir models/xgboost