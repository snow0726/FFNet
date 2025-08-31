import os
import json
import pickle
import torch
from torch.utils.data import Dataset
from typing import Optional
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem import DataStructs
import numpy as np
from typing import Callable, List, Any, Optional, Tuple, Dict, Collection, Union
from tqdm import tqdm

MAX_MZ = 2000
# TODO In hindsight, I should have cleaned the data more here as e.g. M+H probably means [M+H]+, etc.
# Also, M-H is unlikely to be there in positive mode....
ADDUCTS = ['[M+H]+', '[M+Na]+', 'M+H', 'M-H', '[M-H2O+H]+', '[M-H]-', '[M+NH4]+', 'M+NH4', 'M+Na']
RANDOM_SEED = 43242
FINGERPRINT_NBITS = 1024

MAX_ION_SHIFT = 25
FRAGMENT_LEVELS = [-4, -3, -2, -1, 0]
SPECTRA_DIM = MAX_MZ * 2

# 计算增强编码的实际维度
def compute_enhanced_spectra_dim():
    # 使用示例输入计算
    example_spec = [[100.0, 1.0], [200.0, 0.5]]
    encoded, dims = enhanced_spectrum_encoding(example_spec)
    return encoded.shape[0]

# 获取并设置增强编码维度
try:
    ENHANCED_SPECTRA_DIM = compute_enhanced_spectra_dim()
    print(f"计算得到增强谱图编码维度: {ENHANCED_SPECTRA_DIM}")
except:
    # 如果计算失败，使用估计值
    ENHANCED_SPECTRA_DIM = 6552
    print(f"使用估计的增强谱图编码维度: {ENHANCED_SPECTRA_DIM}")


def get_fragmentation_level(mol: Chem.rdchem.Mol, spec: np.array) -> np.array:
    mass = ExactMolWt(mol)
    min_mass, max_mass = int(max(0, mass - MAX_ION_SHIFT)), int(min(mass + MAX_ION_SHIFT, MAX_MZ))
    _spec = 10 ** spec - 1
    frag_level = max(0.01, _spec[min_mass:max_mass + 1].sum())
    return np.log10(frag_level / _spec.sum())


def get_featurized_fragmentation_level(frag_level: int) -> np.array:
    flf = [int(frag_level <= FRAGMENT_LEVELS[0])]
    flf += [int(FRAGMENT_LEVELS[i-1] <= frag_level <= FRAGMENT_LEVELS[i]) for i in range(1, len(FRAGMENT_LEVELS))]
    return np.array(flf)


def get_featurized_adducts(adduct: str) -> np.array:
    return np.array([int(adduct == ADDUCTS[i]) for i in range(len(ADDUCTS))])


def get_featurized_fragmentation_level_from_mol(mol: Chem.rdchem.Mol, spec: np.array) -> np.array:
    fl = get_fragmentation_level(mol, spec)
    return get_featurized_fragmentation_level(fl)


def encode_spec(spec: np.array) -> np.array:
    vec = np.zeros(MAX_MZ * 2)
    for i in range(spec.shape[0]):
        mz_rnd = int(spec[i, 0])
        if mz_rnd >= MAX_MZ:
            continue
        logint= np.log10(spec[i, 1] + 1)
        if vec[mz_rnd] < logint:
            vec[mz_rnd] = logint
            vec[MAX_MZ + mz_rnd] = np.log10((spec[i, 0] - mz_rnd) + 1)
    return vec


def decode_spec(flatspec: np.array, lowest_intensity: float = 0, make_relative: bool = False) -> np.array:
    intensities = flatspec[:len(flatspec) // 2]
    spln = sum(intensities > lowest_intensity)
    spec = np.zeros([spln, 2])
    spec[:, 1] = 10**(intensities[intensities > lowest_intensity]) - 1
    if make_relative:
        spec[:, 1] /= spec[:, 1].max()
    spec[:, 0] = np.where(intensities > lowest_intensity)[0] + (10**(flatspec[len(flatspec) // 2:][intensities > lowest_intensity]) - 1)
    return spec


def fingerprint(
    mol: Chem.rdchem.Mol,
    frag_levels: np.array,
    adduct_feats: np.array,
    nbits: int=FINGERPRINT_NBITS
    ) -> np.array:
    fp = AllChem.GetHashedMorganFingerprint(mol, 2, nBits=nbits)
    mol_rep = np.zeros((0,))
    DataStructs.ConvertToNumpyArray(fp, mol_rep)
    return np.hstack([mol_rep, frag_levels, adduct_feats])

def hunhe_fingerprint1(
        mol: Chem.rdchem.Mol,
        frag_levels: np.array,
        adduct_feats: np.array,
    ) -> np.array:
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    torsion_fp = AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol,nBits=857)
    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    mol_rep = np.zeros((0,))
    torsion_rep = np.zeros((0,))
    mass_rep=np.zeros((0,))
    DataStructs.ConvertToNumpyArray(morgan_fp, mol_rep)
    DataStructs.ConvertToNumpyArray(torsion_fp, torsion_rep)
    DataStructs.ConvertToNumpyArray(maccs_fp, mass_rep)
    return np.hstack([mol_rep, torsion_rep, mass_rep,frag_levels, adduct_feats])

def fingerprint1(
    mol: Chem.rdchem.Mol,
    frag_levels: np.array,
    adduct_feats: np.array,
    nbits: int=FINGERPRINT_NBITS
    ) -> np.array:
    fp = AllChem.GetHashedMorganFingerprint(mol, 2, nBits=nbits)
    mol_rep = np.zeros((0,))
    DataStructs.ConvertToNumpyArray(fp, mol_rep)
    return np.hstack([mol_rep])


def gnps_parser(fname: str, from_mol: int = 0, to_mol: Optional[int] = None) -> List[Any]:
    molecules = []
    spectra = []
    adducts = []
    with open(fname) as fl:
        for i, line in tqdm(list(enumerate(fl.readlines()[from_mol:]))):
            if to_mol is not None and from_mol + i >=  to_mol:
                break
            _, adduct, mol, spec = parse_spectra(line)
            molecules.append(mol)
            spectra.append(spec)
            adducts.append(adduct)
    return molecules, adducts, spectra


def gnps_parser_3d(fnames: Tuple[str, str], from_mol: int = 0, to_mol: Optional[int] = None) -> List[Any]:
    molecules = []
    spectra = []
    adducts = []
    spectra_fname, sdf_fname = fnames

    for i, mol in enumerate(Chem.SDMolSupplier(sdf_fname)):
        if to_mol is not None and from_mol + i >=  to_mol:
            break
        molecules.append(mol)

    with open(spectra_fname) as fl:
        for i, line in tqdm(list(enumerate(fl.readlines()[from_mol:]))):
            if to_mol is not None and from_mol + i >=  to_mol:
                break
            _, adduct, _, spec = parse_spectra(line)
            adducts.append(adduct)
            spectra.append(spec)
    return molecules[:len(spectra)], adducts, spectra

# new

def multi_resolution_encoding(spec_data, max_mz=2000, resolutions=[0.1, 1.0, 5.0]):
    """
    多分辨率谱图编码：在不同精度级别上捕获峰值分布
    
    Args:
        spec_data: 质谱数据 [[mz1, int1], [mz2, int2], ...]
        max_mz: 最大m/z值
        resolutions: 分辨率列表，单位Da
        
    Returns:
        编码后的特征向量
    """
    spec_array = np.array(spec_data)
    mzs = spec_array[:, 0]
    intensities = spec_array[:, 1]
    
    # 归一化强度
    max_intensity = np.max(intensities)
    if max_intensity > 0:
        intensities = intensities / max_intensity
    
    features = []
    
    # 对每个分辨率级别创建特征
    for resolution in resolutions:
        num_bins = int(max_mz / resolution) + 1
        binned_spec = np.zeros(num_bins)
        
        for mz, intensity in zip(mzs, intensities):
            if mz < max_mz:
                bin_idx = int(mz / resolution)
                # 使用最大值聚合而不是简单覆盖
                binned_spec[bin_idx] = max(binned_spec[bin_idx], intensity)
        
        # 对数变换
        binned_spec = np.log1p(binned_spec)
        features.append(binned_spec)
    
    # 将不同分辨率的特征拼接在一起
    return np.concatenate(features)

def peak_clustering_encoding(spec_data, max_mz=2000, resolution=0.1, window_size=1.0):
    """
    峰值聚类编码：将相近的峰值聚集在一起
    
    Args:
        spec_data: 质谱数据 [[mz1, int1], [mz2, int2], ...]
        max_mz: 最大m/z值
        resolution: 最终编码的分辨率
        window_size: 聚类窗口大小
        
    Returns:
        编码后的特征向量
    """
    spec_array = np.array(spec_data)
    if len(spec_array) == 0:
        return np.zeros(int(max_mz / resolution))
        
    mzs = spec_array[:, 0]
    intensities = spec_array[:, 1]
    
    # 归一化强度
    max_intensity = np.max(intensities)
    if max_intensity > 0:
        intensities = intensities / max_intensity
    
    # 按m/z值排序
    sorted_indices = np.argsort(mzs)
    mzs = mzs[sorted_indices]
    intensities = intensities[sorted_indices]
    
    # 聚类峰值
    clusters = []
    current_cluster = []
    
    for i, (mz, intensity) in enumerate(zip(mzs, intensities)):
        if not current_cluster or mz - current_cluster[-1][0] <= window_size:
            current_cluster.append((mz, intensity))
        else:
            clusters.append(current_cluster)
            current_cluster = [(mz, intensity)]
    
    if current_cluster:
        clusters.append(current_cluster)
    
    # 创建编码向量
    num_bins = int(max_mz / resolution)
    encoded_spec = np.zeros(num_bins)
    
    # 处理每个聚类
    for cluster in clusters:
        cluster_mzs = np.array([p[0] for p in cluster])
        cluster_intensities = np.array([p[1] for p in cluster])
        
        # 计算强度加权的中心m/z值
        if np.sum(cluster_intensities) > 0:
            weighted_mz = np.sum(cluster_mzs * cluster_intensities) / np.sum(cluster_intensities)
        else:
            weighted_mz = np.mean(cluster_mzs)
            
        # 计算聚类的总强度
        total_intensity = np.sum(cluster_intensities)
        
        # 将加权m/z值映射到对应的bin
        bin_idx = int(weighted_mz / resolution)
        if 0 <= bin_idx < num_bins:
            encoded_spec[bin_idx] = total_intensity
    
    # 对数变换
    encoded_spec = np.log1p(encoded_spec)
    
    return encoded_spec

def peak_relationship_features(spec_data, top_n=30):
    """
    提取峰值间关系特征
    
    Args:
        spec_data: 质谱数据 [[mz1, int1], [mz2, int2], ...]
        top_n: 考虑的最高峰值数量
        
    Returns:
        关系特征向量
    """
    spec_array = np.array(spec_data)
    if len(spec_array) <= 1:
        # 如果峰值数量太少，返回全零特征
        return np.zeros(50)
        
    mzs = spec_array[:, 0]
    intensities = spec_array[:, 1]
    
    # 选择top_n峰值
    top_indices = np.argsort(intensities)[-top_n:]
    top_mzs = mzs[top_indices]
    top_intensities = intensities[top_indices]
    
    # 计算峰值间距特征
    features = []
    
    # 1. 质量差特征
    if len(top_mzs) >= 2:
        sorted_mzs = np.sort(top_mzs)
        mass_diffs = sorted_mzs[1:] - sorted_mzs[:-1]
        
        # 添加常见的质量差计数
        common_diffs = [1.0, 2.0, 14.0, 16.0, 18.0, 28.0, 30.0, 44.0]  # 常见碎片质量差
        for diff in common_diffs:
            features.append(np.sum((mass_diffs >= diff - 0.5) & (mass_diffs <= diff + 0.5)))
        
        # 质量差统计特征
        features.extend([
            np.min(mass_diffs),
            np.max(mass_diffs),
            np.mean(mass_diffs),
            np.median(mass_diffs),
            np.std(mass_diffs)
        ])
    else:
        # 填充零值
        features.extend([0] * (len(common_diffs) + 5))
    
    # 2. 强度比特征
    if len(top_intensities) >= 2:
        sorted_intensities = np.sort(top_intensities)
        intensity_ratios = [
            sorted_intensities[-1] / (sorted_intensities[0] + 1e-10),  # 最大/最小强度比
            sorted_intensities[-1] / (np.mean(sorted_intensities) + 1e-10),  # 最大/平均强度比
            np.mean(sorted_intensities[len(sorted_intensities)//2:]) / (np.mean(sorted_intensities[:len(sorted_intensities)//2]) + 1e-10)  # 上半部分/下半部分平均强度比
        ]
        features.extend(intensity_ratios)
    else:
        features.extend([0, 0, 0])
    
    # 3. 峰值分布特征
    mz_ranges = [0, 100, 200, 300, 500, 800, 1500, 2000]
    for i in range(len(mz_ranges)-1):
        min_mz, max_mz = mz_ranges[i], mz_ranges[i+1]
        peaks_in_range = np.sum((mzs >= min_mz) & (mzs < max_mz))
        intensity_in_range = np.sum(intensities[(mzs >= min_mz) & (mzs < max_mz)])
        
        features.extend([peaks_in_range, intensity_in_range])
    
    # 确保特征向量长度固定
    if len(features) < 50:
        features.extend([0] * (50 - len(features)))
    else:
        features = features[:50]
    
    return np.array(features)

def extract_top_n_peaks(spec_data, n=50):
    """
    提取前N个最强峰值的特征
    
    Args:
        spec_data: 质谱数据 [[mz1, int1], [mz2, int2], ...]
        n: 提取的峰值数量
        
    Returns:
        TOP-N峰值特征向量
    """
    spec_array = np.array(spec_data)
    if len(spec_array) == 0:
        return np.zeros(n * 2)
        
    mzs = spec_array[:, 0]
    intensities = spec_array[:, 1]
    
    # 按强度排序并选择前N个
    sorted_indices = np.argsort(intensities)[::-1][:n]
    top_mzs = mzs[sorted_indices]
    top_intensities = intensities[sorted_indices]
    
    # 归一化强度
    max_intensity = np.max(top_intensities) if len(top_intensities) > 0 else 1.0
    top_intensities = top_intensities / max_intensity if max_intensity > 0 else top_intensities
    
    # 创建特征向量
    features = np.zeros(n * 2)
    
    # 填充特征向量
    for i in range(min(n, len(top_mzs))):
        features[i] = top_mzs[i] / 2000.0  # 归一化m/z值
        features[i + n] = top_intensities[i]
    
    return features

def enhanced_spectrum_encoding(spec_data, max_mz=2000):
    """
    增强型谱图编码：结合多种策略
    
    Args:
        spec_data: 质谱数据 [[mz1, int1], [mz2, int2], ...]
        max_mz: 最大m/z值
        
    Returns:
        编码后的特征向量和特征维度说明
    """
    # 1. 多分辨率编码
    resolutions = [1.0, 5.0]  # 使用两种分辨率级别
    multi_res_features = multi_resolution_encoding(spec_data, max_mz, resolutions)
    
    # 2. 峰值聚类编码
    cluster_features = peak_clustering_encoding(spec_data, max_mz, resolution=0.5, window_size=0.5)
    
    # 3. 峰值关系特征
    relationship_features = peak_relationship_features(spec_data, top_n=30)
    
    # 4. TOP-N峰值直接编码
    top_n = 50
    top_features = extract_top_n_peaks(spec_data, top_n)
    
    # 合并所有特征
    all_features = np.concatenate([
        multi_res_features,  
        cluster_features,
        relationship_features,
        top_features
    ])
    
    # 返回特征向量和各部分维度说明
    feature_dims = {
        'multi_resolution': multi_res_features.shape[0],
        'peak_clustering': cluster_features.shape[0],
        'peak_relationships': relationship_features.shape[0],
        'top_n_peaks': top_features.shape[0],
        'total': all_features.shape[0]
    }
    
    return all_features, feature_dims

def enhanced_encode_spec(spec_data, max_mz=2000):
    """
    增强型质谱编码，替代原始的encode_spec函数
    
    Args:
        spec_data: 质谱数据，列表形式的峰值 [[m/z1, int1], [m/z2, int2], ...]
        max_mz: 最大m/z值
        
    Returns:
        编码后的特征向量
    """
    # 使用综合编码策略
    features, _ = enhanced_spectrum_encoding(spec_data, max_mz)
    return features

def parse_enhanced_spectra(line: str) -> Tuple[str, str, Chem.rdchem.Mol, np.array]:
    """使用增强编码策略解析谱图数据"""
    sid, adduct, spec_str, smiles = line.strip().split('\t')
    spec_data = np.array(json.loads(spec_str))
    
    if not(len(spec_data.shape) == 2 and spec_data.shape[0] >= 1 and (spec_data <= 0).sum() == 0):
        print(spec_data)
        raise ValueError('Invalid spectrum data')
    
    # 使用增强型编码
    encoded_spec, _ = enhanced_spectrum_encoding(spec_data, max_mz=MAX_MZ)
    mol = Chem.MolFromSmiles(smiles)
    
    return sid, adduct, mol, encoded_spec

def enhanced_gnps_parser(fname: str, from_mol: int = 0, to_mol: Optional[int] = None) -> List[Any]:
    """增强型谱图解析函数，使用改进的编码策略"""
    molecules = []
    spectra = []
    adducts = []
    with open(fname) as fl:
        for i, line in tqdm(list(enumerate(fl.readlines()[from_mol:]))):
            if to_mol is not None and from_mol + i >=  to_mol:
                break
            _, adduct, mol, spec = parse_enhanced_spectra(line)
            molecules.append(mol)
            spectra.append(spec)
            adducts.append(adduct)
    return molecules, adducts, spectra



# new


def parse_spectra(line: str) -> Tuple[str, str, Chem.rdchem.Mol, np.array]:
    sid, adduct, spec_str, smiles = line.strip().split('\t')
    spec = np.array(json.loads(spec_str))
    if not( len(spec.shape) == 2 and spec.shape[0] >= 1 and (spec <= 0).sum() == 0):
        print(spec)
        raise ValueError('what')
    # Round to 3 digits M/Z precision
    spec[:, 0] = np.round(spec[:, 0], 3)
    # We'll predict relative intensities
    spec[:, 1] /= spec[:, 1].max()
    mol = Chem.MolFromSmiles(smiles)
    return sid, adduct, mol, encode_spec(spec)


class Spec2MolDataset(Dataset):
    SAVED_PROPS = [
            'molecules',
            'spectra',
            'frag_levels',
            'adducts',
            'adduct_feats',
            'mol_reps',
    ]

    SAVE_DIR = '/data/zhangxiaofeng/code/code/data/data'

    def __init__(
        self,
        dataset_name: str,
        fnames: Union[str, Collection[str]],
        parser: Callable,
        mol_representation: Callable = fingerprint,
        from_mol: int = 0,
        to_mol: Optional[int] = None,
        use_cache: bool = False,
        mol_rep_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.dataset_name = dataset_name
        if use_cache and self.is_cached():
            print('Cache found, loading dataset')
            self.load()
        else:
            self.molecules, self.adducts, self.spectra = parser(fnames, from_mol=from_mol, to_mol=to_mol)
            self.frag_levels = [get_featurized_fragmentation_level_from_mol(mol, spec) for mol, spec in zip(self.molecules, self.spectra)]
            self.adduct_feats = [get_featurized_adducts(adduct) for adduct in self.adducts]
            self.mol_representation = mol_representation
            self.mol_rep_kwargs = mol_rep_kwargs if mol_rep_kwargs is not None else {}

            self.mol_reps = [
                self.mol_representation(mol, frag_levels=self.frag_levels[i], adduct_feats=self.adduct_feats[i], **self.mol_rep_kwargs)
                for i, mol in tqdm(list(enumerate(self.molecules)), desc='Calculating mol reps')
                ]
            if use_cache:
                print('Caching dataset')
                self.save()

    @property
    def cache_fname(self):
        return os.path.join(Spec2MolDataset.SAVE_DIR, self.dataset_name + '.pkl')

    def is_cached(self):
        return os.path.exists(self.cache_fname)

    def save(self):
        with open(self.cache_fname, 'wb') as fl:
            pickle.dump({k: getattr(self, k) for k in Spec2MolDataset.SAVED_PROPS}, fl)
            # Weird torch_geometric bug that needs to reload pickled object to regenerate globalstorage
            for k in Spec2MolDataset.SAVED_PROPS:
                delattr(self, k)
            self.load()

    def load(self):
        with open(self.cache_fname, 'rb') as fl:
            props = pickle.load(fl)
            for k, v in props.items():
                setattr(self, k, v)

    def __len__(self):
        return len(self.mol_reps)

    def __getitem__(self, idx: int):
            return self.spectra[idx], torch.FloatTensor(self.mol_reps[idx])


class identificationDataset(Dataset):
    SAVED_PROPS = [
            'molecules',
            'spectra',
            'frag_levels',
            'adducts',
            'adduct_feats',
            'mol_reps',
            'precursor_mz',
    ]

    SAVE_DIR = 'data'

    def __init__(
        self,
        dataset_name: str,
        fnames: Union[str, Collection[str]],
        parser: Callable,
        mol_representation: Callable = fingerprint,
        from_mol: int = 0,
        to_mol: Optional[int] = None,
        use_cache: bool = False,
        mol_rep_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.dataset_name = dataset_name
        if use_cache and self.is_cached():
            print('Cache found, loading dataset')
            self.load()
        else:
            self.molecules, self.adducts, self.spectra, self.precursor_mz = parser(fnames, from_mol=from_mol, to_mol=to_mol)
            self.frag_levels = [get_featurized_fragmentation_level_from_mol(mol, spec) for mol, spec in zip(self.molecules, self.spectra)]
            self.adduct_feats = [get_featurized_adducts(adduct) for adduct in self.adducts]
            self.mol_representation = mol_representation
            self.smiles = [Chem.MolToSmiles(mol) for mol in self.molecules]
            self.mol_rep_kwargs = mol_rep_kwargs if mol_rep_kwargs is not None else {}

            self.mol_reps = [
                self.mol_representation(mol, frag_levels=self.frag_levels[i], adduct_feats=self.adduct_feats[i], **self.mol_rep_kwargs)
                for i, mol in tqdm(list(enumerate(self.molecules)), desc='Calculating mol reps')
                ]
            if use_cache:
                print('Caching dataset')
                self.save()

    @property
    def cache_fname(self):
        return os.path.join(identificationDataset.SAVE_DIR, self.dataset_name + '.pkl')

    def is_cached(self):
        return os.path.exists(self.cache_fname)

    def save(self):
        with open(self.cache_fname, 'wb') as fl:
            pickle.dump({k: getattr(self, k) for k in identificationDataset.SAVED_PROPS}, fl)
            # Weird torch_geometric bug that needs to reload pickled object to regenerate globalstorage
            for k in identificationDataset.SAVED_PROPS:
                delattr(self, k)
            self.load()

    def load(self):
        with open(self.cache_fname, 'rb') as fl:
            props = pickle.load(fl)
            for k, v in props.items():
                setattr(self, k, v)

    def __len__(self):
        return len(self.mol_reps)

    def __getitem__(self, idx: int):
            return self.spectra[idx], torch.FloatTensor(self.mol_reps[idx]), self.precursor_mz[idx], self.smiles[idx]

