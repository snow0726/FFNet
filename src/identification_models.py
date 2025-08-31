#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
质谱鉴定程序 - 使用模型预测的分子指纹进行候选排序
作者: snow0726
日期: 2025-03-19 02:13:34
描述: 使用训练好的模型对未知质谱进行分析,预测分子指纹,并与PubChem库中的化合物进行比较
"""

import os
import json
import time
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem import DataStructs
from rdkit.Chem.Descriptors import ExactMolWt
from scipy.spatial.distance import cosine
import argparse
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Tuple, Set, Any, Optional, Callable, Union, Collection
from tqdm import tqdm
import sys
import utils, models

# 常量定义
# PROTON_MASS = 1.007825  # 质子质量(Da)
PROTON_MASS = 0
MZ_TOLERANCE = 0.01     # m/z匹配容差(Da)
INTENSITY_NOISE_THRESHOLD = 0.01  # 强度阈值，低于此值被视为噪声
MAX_MZ = 2000  # 最大m/z值
ADDUCTS = ['[M+H]+', '[M+Na]+', 'M+H', 'M-H', '[M-H2O+H]+', '[M-H]-', '[M+NH4]+', 'M+NH4', 'M+Na']
RANDOM_SEED = 43242
FINGERPRINT_NBITS = 1024
MAX_ION_SHIFT = 25
FRAGMENT_LEVELS = [-4, -3, -2, -1, 0]
SPECTRA_DIM = MAX_MZ * 2

# 设置随机种子
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# 重用您提供的函数
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
    """将质谱峰列表转换为模型输入向量格式"""
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
    """将模型输入格式的谱图转换回峰列表"""
    intensities = flatspec[:len(flatspec) // 2]
    spln = sum(intensities > lowest_intensity)
    spec = np.zeros([spln, 2])
    spec[:, 1] = 10**(intensities[intensities > lowest_intensity]) - 1
    if make_relative:
        spec[:, 1] /= spec[:, 1].max()
    spec[:, 0] = np.where(intensities > lowest_intensity)[0] + (10**(flatspec[len(flatspec) // 2:][intensities > lowest_intensity]) - 1)
    return spec

def hunhe_fingerprint1(
        mol: Chem.rdchem.Mol,
        frag_levels: np.array,
        adduct_feats: np.array,
    ) -> np.array:
    """计算混合分子指纹"""
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    torsion_fp = AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=857)
    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    mol_rep = np.zeros((0,))
    torsion_rep = np.zeros((0,))
    mass_rep=np.zeros((0,))
    DataStructs.ConvertToNumpyArray(morgan_fp, mol_rep)
    DataStructs.ConvertToNumpyArray(torsion_fp, torsion_rep)
    DataStructs.ConvertToNumpyArray(maccs_fp, mass_rep)
    return np.hstack([mol_rep, torsion_rep, mass_rep, frag_levels, adduct_feats])

def parse_spectra(line: str) -> Tuple[str, str, Chem.rdchem.Mol, np.array]:
    """解析谱图数据行"""
    parts = line.strip().split('\t')
    
    if len(parts) >= 6:  # 鉴定数据集格式
        sid, inchikey, smiles, adduct, precursor_mz, spec_str = parts[:6]
    elif len(parts) >= 4:  # 训练数据集格式
        sid, adduct, spec_str, smiles = parts
    else:
        raise ValueError(f"无效的数据格式: {line[:50]}...")
    
    try:
        spec = np.array(json.loads(spec_str))
    except:
        raise ValueError(f"谱图JSON解析错误: {spec_str[:50]}...")
    
    if not (len(spec.shape) == 2 and spec.shape[0] >= 1 and (spec <= 0).sum() == 0):
        raise ValueError(f"无效的谱图格式: {spec}")
    
    # 四舍五入到3位小数精度
    spec[:, 0] = np.round(spec[:, 0], 3)
    
    # 归一化强度
    spec[:, 1] /= spec[:, 1].max()
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无效的SMILES: {smiles}")
    
    return sid, adduct, mol, encode_spec(spec)

class IdentificationDataset(Dataset):
    """用于鉴定的数据集类"""
    
    def __init__(self, spectra_data, adduct_type='[M+H]+'):
        self.spectra = []
        self.ids = []
        self.inchikeys = []
        self.adducts = []
        
        # 确保所有数据都是[M+H]+
        for spec in spectra_data:
            if spec['adduct'] == adduct_type:
                self.spectra.append(torch.FloatTensor(spec['encoded_spectrum']))
                self.ids.append(spec['id'])
                self.inchikeys.append(spec['inchikey'])
                self.adducts.append(spec['adduct'])
    
    def __len__(self):
        return len(self.spectra)
    
    def __getitem__(self, idx):
        return self.spectra[idx], self.ids[idx], self.inchikeys[idx], self.adducts[idx]

class PubChemDatabase:
    """PubChem化合物数据库类"""
    
    def __init__(self, cid_inchikey_file, cid_mass_file, cid_smiles_file, db_file='/data/zhangxiaofeng/code/code/spectomol/pubchem.db'):
        """初始化PubChem数据库，使用SQLite而不是内存字典"""
        import sqlite3
        
        print("正在初始化PubChem数据库...")
        
        # 创建数据库连接
        self.db_file = db_file
        create_db = not os.path.exists(db_file)
        
        self.conn = sqlite3.connect(db_file)
        self.cursor = self.conn.cursor()
        
        # 如果是新数据库，创建表结构
        if create_db:
            print("创建新的PubChem数据库...")
            
            # 创建表结构
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS compounds (
                cid TEXT PRIMARY KEY,
                inchikey TEXT,
                smiles TEXT,
                mass REAL
            )''')
            
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_inchikey ON compounds(inchikey)')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_mass ON compounds(mass)')
            
            # 创建缓存表
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS fingerprints (
                cid TEXT PRIMARY KEY,
                fingerprint BLOB
            )''')
            
            # 加载数据
            print("加载CID-InChIKey映射...")
            with open(cid_inchikey_file, 'r', encoding='utf-8') as f:
                batch = []
                for i, line in enumerate(f):
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        cid = parts[0]
                        inchikey = parts[2]
                        batch.append((cid, inchikey, None, None))
                        
                        if len(batch) >= 10000000:
                            self.cursor.executemany(
                                'INSERT OR IGNORE INTO compounds (cid, inchikey, smiles, mass) VALUES (?, ?, ?, ?)', 
                                batch
                            )
                            self.conn.commit()
                            batch = []
                            print(f"处理了 {i+1} 行...")
                
                if batch:
                    self.cursor.executemany(
                        'INSERT OR IGNORE INTO compounds (cid, inchikey, smiles, mass) VALUES (?, ?, ?, ?)', 
                        batch
                    )
                    self.conn.commit()
            
            # 加载CID-Mass映射
            print("加载CID-Mass映射...")
            with open(cid_mass_file, 'r', encoding='utf-8') as f:
                batch = []
                for i, line in enumerate(f):
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        cid = parts[0]
                        try:
                            mono_mass = float(parts[2])
                            batch.append((mono_mass, cid))
                            
                            if len(batch) >= 10000000:
                                self.cursor.executemany(
                                    'UPDATE compounds SET mass = ? WHERE cid = ?',
                                    batch
                                )
                                self.conn.commit()
                                batch = []
                                print(f"处理了 {i+1} 行...")
                        except (ValueError, IndexError):
                            continue
                
                if batch:
                    self.cursor.executemany(
                        'UPDATE compounds SET mass = ? WHERE cid = ?',
                        batch
                    )
                    self.conn.commit()
            
            # 加载CID-SMILES映射
            print("加载CID-SMILES映射...")
            with open(cid_smiles_file, 'r', encoding='utf-8') as f:
                batch = []
                for i, line in enumerate(f):
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        cid = parts[0]
                        smiles = parts[1]
                        batch.append((smiles, cid))
                        
                        if len(batch) >= 10000000:
                            self.cursor.executemany(
                                'UPDATE compounds SET smiles = ? WHERE cid = ?',
                                batch
                            )
                            self.conn.commit()
                            batch = []
                            print(f"处理了 {i+1} 行...")
                
                if batch:
                    self.cursor.executemany(
                        'UPDATE compounds SET smiles = ? WHERE cid = ?',
                        batch
                    )
                    self.conn.commit()
            
            # 创建质量索引
            print("创建索引...")
            self.conn.commit()
        
        # 获取数据库大小
        count = self.cursor.execute("SELECT COUNT(*) FROM compounds").fetchone()[0]
        print(f"PubChem数据库加载完成: {count} 条记录")
        
        # 仅保留少量的内存中指纹缓存
        self.fingerprints_cache = {}
        self.cache_size_limit = 10000000  # 限制内存中的缓存大小
        
    def find_candidates_by_mass(self, target_mass, tolerance=0.01, max_candidates=100):
        """根据质量查找候选化合物，使用SQL查询而不是内存操作"""
        min_mass = target_mass - tolerance
        max_mass = target_mass + tolerance
        
        # 使用SQL查询找出质量范围内的所有化合物
        self.cursor.execute(
            """
            SELECT cid, inchikey, smiles, mass
            FROM compounds
            WHERE mass BETWEEN ? AND ?
            ORDER BY ABS(mass - ?)
            LIMIT ?
            """,
            (min_mass, max_mass, target_mass, max_candidates)
        )
        
        candidates = []
        for cid, inchikey, smiles, mass in self.cursor.fetchall():
            if inchikey:
                candidates.append({
                    'cid': cid,
                    'inchikey': inchikey,
                    'mass': mass,
                    'mass_error': mass - target_mass if mass else 0,
                    'smiles': smiles
                })
        
        return candidates
    
    def get_compound_fingerprint(self, cid, adduct_type='[M+H]+'):
        """获取化合物的指纹向量，增加从数据库加载指纹的功能"""
        # 先检查内存缓存
        if cid in self.fingerprints_cache:
            return self.fingerprints_cache[cid]
        
        # 检查数据库缓存
        self.cursor.execute('SELECT fingerprint FROM fingerprints WHERE cid = ?', (cid,))
        result = self.cursor.fetchone()
        if result:
            import pickle
            fingerprint = pickle.loads(result[0])
            
            # 更新内存缓存，确保不超过限制
            if len(self.fingerprints_cache) >= self.cache_size_limit:
                # 随机删除一项以控制大小
                import random
                del_key = random.choice(list(self.fingerprints_cache.keys()))
                del self.fingerprints_cache[del_key]
            
            self.fingerprints_cache[cid] = fingerprint
            return fingerprint
        
        # 如果没有缓存，从SMILES计算
        self.cursor.execute('SELECT smiles FROM compounds WHERE cid = ?', (cid,))
        result = self.cursor.fetchone()
        if not result or not result[0]:
            return None
        
        smiles = result[0]
        
        # 以下代码与原始版本相同，但增加了保存到数据库的功能
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
        
        # 创建一个假的谱图用于计算fragmentation level
        dummy_spec = np.zeros(SPECTRA_DIM)
        frag_levels = get_featurized_fragmentation_level_from_mol(mol, dummy_spec)
        adduct_feats = get_featurized_adducts(adduct_type)
        
        # 计算混合指纹
        fingerprint = hunhe_fingerprint1(mol, frag_levels, adduct_feats)
        
        # 保存到数据库缓存
        import pickle
        blob = pickle.dumps(fingerprint)
        self.cursor.execute(
            'INSERT OR REPLACE INTO fingerprints (cid, fingerprint) VALUES (?, ?)',
            (cid, blob)
        )
        self.conn.commit()
        
        # 更新内存缓存，确保不超过限制
        if len(self.fingerprints_cache) >= self.cache_size_limit:
            # 随机删除一项以控制大小
            import random
            del_key = random.choice(list(self.fingerprints_cache.keys()))
            del self.fingerprints_cache[del_key]
        
        self.fingerprints_cache[cid] = fingerprint
        return fingerprint

def load_model(model_path, model_type=None):
    """加载预训练模型"""
    print(f"加载模型: {model_path}")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 如果没有指定模型类型，则尝试从路径检测
    if model_type is None:
        if "mlp" in model_path.lower():
            model_type = "mlp"
        elif "transformer" in model_path.lower():
            model_type = "transformer"
        else:
            model_type = "featurefusion"  # 默认值
    
    print(f"使用模型类型: {model_type}")
    
    # 根据模型类型创建相应的模型实例
    if model_type == "featurefusion":
        model = models.Spec2MolFeaturefusion(
            molecule_dim=utils.SPECTRA_DIM,
            prop_dim=utils.FINGERPRINT_NBITS+857+167+len(utils.FRAGMENT_LEVELS)+len(utils.ADDUCTS),
            hdim=2048,  
            n_layers=6  
        )
    elif model_type == "mlp":
        model = models.Mol2SpecSimple(
            molecule_dim=utils.SPECTRA_DIM,
            prop_dim=utils.FINGERPRINT_NBITS+857+167+len(utils.FRAGMENT_LEVELS)+len(utils.ADDUCTS),
            hdim=1024,  
            n_layers=4  
        )
    elif model_type == "transformer":
        model = models.MyEncoder(
            input_dim=utils.SPECTRA_DIM,
            output_dim=utils.FINGERPRINT_NBITS+857+167+len(utils.FRAGMENT_LEVELS)+len(utils.ADDUCTS),
            hidden_dim=1024,  
            num_layers=4, 
            num_heads=8
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 加载预训练权重
    state_dict = torch.load(model_path, map_location=device)
    
    # 检查并提取模型状态字典
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    
    # 处理可能的DataParallel包装
    if list(state_dict.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # 移除 'module.' 前缀
            new_state_dict[name] = v
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"模型已加载至设备: {device}")
    return model, device

def load_identification_dataset(file_path):
    """加载鉴定数据集"""
    print(f"加载鉴定数据集: {file_path}")
    spectra_data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 6:
                try:
                    spectrum_id = parts[0]
                    inchikey = parts[1]
                    smiles = parts[2]
                    adduct = parts[3]
                    precursor_mz = float(parts[4])
                    peaks = json.loads(parts[5])
                    
                    # 确保adduct是[M+H]+
                    if adduct == "[M+H]+":
                        # 计算分子质量
                        neutral_mass = precursor_mz - PROTON_MASS
                        
                        # 将峰列表转换为numpy数组
                        peaks_array = np.array(peaks)
                        
                        # 归一化强度
                        if peaks_array.shape[0] > 0:
                            peaks_array[:, 1] = peaks_array[:, 1] / peaks_array[:, 1].max()
                        
                        # 编码谱图
                        encoded_spectrum = encode_spec(peaks_array)
                        
                        spectra_data.append({
                            'id': spectrum_id,
                            'inchikey': inchikey,
                            'smiles': smiles,
                            'adduct': adduct,
                            'precursor_mz': precursor_mz,
                            'neutral_mass': neutral_mass,
                            'peaks': peaks_array,
                            'encoded_spectrum': encoded_spectrum
                        })
                except Exception as e:
                    print(f"解析行时出错: {e}, 行: {line[:50]}...")
                    continue
    
    print(f"成功加载 {len(spectra_data)} 条谱图记录")
    return spectra_data

def load_reference_library(file_path):
    """加载参考库"""
    print(f"加载参考库: {file_path}")
    references = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # 跳过标题行
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 6:
                try:
                    spectrum_id = parts[0]
                    inchikey = parts[1]
                    name = parts[2]
                    smiles = parts[3]
                    mw = float(parts[4]) if parts[4] else None
                    formula = parts[5]
                    source = parts[6] if len(parts) > 6 else "Unknown"
                    
                    references[inchikey] = {
                        'id': spectrum_id,
                        'name': name,
                        'smiles': smiles,
                        'molecular_weight': mw,
                        'formula': formula,
                        'source': source
                    }
                except Exception as e:
                    print(f"解析参考库行时出错: {e}, 行: {line[:50]}...")
                    continue
    
    print(f"成功加载 {len(references)} 条参考记录")
    return references

def predict_fingerprints(model, device, dataset, batch_size=32):
    """使用模型预测分子指纹"""
    print("开始预测分子指纹...")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_ids = []
    all_inchikeys = []
    
    with torch.no_grad():
        for spectra, ids, inchikeys, _ in tqdm(dataloader, desc="预测分子指纹"):
            # 将数据移至GPU
            spectra = spectra.to(device)
            
            # 模型预测
            predictions = model(spectra)  # 您的模型直接输出是指纹向量
            
            # 将结果移至CPU并转换为numpy
            predictions = predictions.cpu().numpy()
            
            # 收集结果
            all_predictions.extend(predictions)
            all_ids.extend(ids)
            all_inchikeys.extend(inchikeys)
    
    # 将结果整合为字典
    results = {
        'predictions': np.array(all_predictions),
        'ids': all_ids,
        'inchikeys': all_inchikeys
    }
    
    print(f"完成 {len(all_ids)} 条谱图的指纹预测")
    return results

def calculate_similarity(pred_fp, candidate_fp):
    """计算两个指纹向量的余弦相似度"""
    similarity = 1 - cosine(pred_fp, candidate_fp)
    return similarity

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='质谱鉴定程序 - 基于指纹预测')
    
    # 输入文件
    parser.add_argument('--ident_dataset', type=str, required=True,
                        help='鉴定数据集TSV文件路径')
    parser.add_argument('--reference_library', type=str, required=True,
                        help='参考库TSV文件路径')
    parser.add_argument('--cid_inchikey', type=str, required=True,
                        help='CID-InChI-Key文件路径')
    parser.add_argument('--cid_mass', type=str, required=True,
                        help='CID-Mass文件路径')
    parser.add_argument('--cid_smiles', type=str, required=True,
                        help='CID-SMILES文件路径')
    parser.add_argument('--model_path', type=str, required=True,
                        help='预训练模型路径')
    parser.add_argument('--model_type', type=str, default=None,
                    help='Model type: "featurefusion", "mlp", or "transformer". If not specified, will try to detect from model path.')
    parser.add_argument('--eval_batch_size', type=int, default=10240,
                        help='鉴定评估时的批处理大小，用于控制内存使用')
    
    # 参数设置
    parser.add_argument('--mass_tolerance', type=float, default=0.01,
                        help='质量匹配容差(Da)，默认为0.01')
    parser.add_argument('--top_n', type=str, default='1,3,5,10,20',
                        help='评估的Top-N值，用逗号分隔，默认为1,3,5,10,20')
    parser.add_argument('--output_dir', type=str, default='./identification_results',
                        help='输出目录，默认为./identification_results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批处理大小，默认为32')
    parser.add_argument('--max_candidates', type=int, default=100,
                        help='每个谱图的最大候选化合物数，默认为100')
    parser.add_argument('--cache_fingerprints', action='store_true', 
                        help='是否缓存PubChem指纹计算结果')
    parser.add_argument('--cache_file', type=str, default='./pubchem_fingerprints.pkl',
                        help='PubChem指纹缓存文件路径')
    
    return parser.parse_args()

def generate_detailed_report(results, reference_library, output_dir):
    """生成详细的鉴定报告，包含参考库信息"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"detailed_report_{timestamp}.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=== 质谱鉴定详细报告 ===\n\n")
        f.write(f"总谱图数: {results['total']}\n")
        f.write(f"日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Top-N 准确率:\n")
        for n in sorted([int(k.split('_')[-1]) for k in results.keys() if k.startswith('accuracy_top_')]):
            accuracy = results[f'accuracy_top_{n}']
            f.write(f"- Top-{n}: {accuracy:.4f} ({results['correct'][n]}/{results['total']})\n")
        
        f.write("\n=== 详细结果 ===\n\n")
        
        for i, detail in enumerate(results['details']):
            f.write(f"谱图 #{i+1}: {detail['spectrum_id']}\n")
            
            # 获取真实化合物的信息
            true_inchikey = detail['true_inchikey']
            true_info = detail.get('true_compound_info', {})
            
            if true_info:
                f.write(f"正确化合物: {true_info.get('name', '未知')} (InChIKey: {true_inchikey})\n")
                f.write(f"- 分子量: {true_info.get('molecular_weight', '未知')}\n")
                f.write(f"- 分子式: {true_info.get('formula', '未知')}\n")
                f.write(f"- 来源: {true_info.get('source', '未知')}\n")
            else:
                f.write(f"正确化合物: InChIKey: {true_inchikey} (参考库中未找到)\n")
            
            # 输出正确候选排名
            if 'correct_rank' in detail:
                f.write(f"正确候选排名: {detail['correct_rank']}\n")
            else:
                f.write("正确候选排名: 未在候选列表中\n")
            
            # 输出前5个候选
            f.write(f"前 {min(5, detail['candidate_count'])} 个候选化合物:\n")
            for cand in detail['top_candidates']:
                f.write(f"- 排名 {cand['rank']}: {cand['inchikey']}")
                
                # 从参考库获取候选信息
                ref_info = cand.get('reference_info', {})
                if ref_info:
                    f.write(f" ({ref_info.get('name', '未知')})")
                
                f.write(f", 相似度: {cand['similarity']:.4f}")
                f.write(f", 质量误差: {cand['mass_error']:.6f} Da")
                
                if cand['is_correct']:
                    f.write(" [正确]")
                f.write("\n")
            
            f.write("\n" + "-"*50 + "\n\n")
    
    print(f"详细报告已保存至: {report_file}")

def identify_compound(spectrum_data, pred_fingerprint, pubchem_db, mass_tolerance=0.01, max_candidates=100):
    """使用预测的指纹识别化合物"""
    # 计算中性分子质量
    neutral_mass = spectrum_data['neutral_mass']
    
    # 查找候选化合物
    candidates = pubchem_db.find_candidates_by_mass(neutral_mass, mass_tolerance, max_candidates)
    
    if not candidates:
        return []
    
    # 计算每个候选化合物与预测指纹的相似度
    for candidate in candidates:
        cid = candidate['cid']
        # 获取候选化合物的指纹
        candidate_fp = pubchem_db.get_compound_fingerprint(cid)
        
        if candidate_fp is None:
            # 如果无法计算指纹，则给予低相似度
            candidate['similarity'] = 0.0
        else:
            # 计算相似度
            similarity = calculate_similarity(pred_fingerprint, candidate_fp)
            candidate['similarity'] = similarity
    
    # 按相似度排序
    sorted_candidates = sorted(candidates, key=lambda x: x['similarity'], reverse=True)
    
    # 添加排名
    for i, candidate in enumerate(sorted_candidates):
        candidate['rank'] = i + 1
    
    return sorted_candidates

def evaluate_identification(spectra_data, pred_results, pubchem_db, reference_library, 
                           top_n_list=[1, 3, 5, 10, 20], 
                           mass_tolerance=0.01,
                           max_candidates=100,
                           output_dir='./identification_results',
                           batch_size=10240):  # 新增batch_size参数
    """评估鉴定准确率，分批处理以减少内存使用"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备结果统计
    results = {
        'total': len(spectra_data),
        'correct': {n: 0 for n in top_n_list},
        'details': []
    }
    
    print(f"开始评估 {len(spectra_data)} 条谱图的鉴定结果...")
    
    # 将预测结果转换为字典，方便查找
    pred_dict = {id_: pred for id_, pred in zip(pred_results['ids'], pred_results['predictions'])}
    
    # 分批处理谱图
    for batch_start in range(0, len(spectra_data), batch_size):
        batch_end = min(batch_start + batch_size, len(spectra_data))
        batch = spectra_data[batch_start:batch_end]
        
        print(f"处理批次 {batch_start//batch_size + 1}/{(len(spectra_data)-1)//batch_size + 1} ({batch_start+1}-{batch_end}/{len(spectra_data)})...")
        
        # 写入进度文件，方便监控
        with open(os.path.join(output_dir, "progress.txt"), "w") as f:
            f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"处理进度: {batch_end}/{len(spectra_data)} ({batch_end/len(spectra_data)*100:.2f}%)\n")
        
        # 对批次中的每个谱图进行鉴定
        for i, spectrum in enumerate(batch):
            spectrum_id = spectrum['id']
            
            # 获取预测的指纹
            if spectrum_id not in pred_dict:
                print(f"警告: 找不到谱图 {spectrum_id} 的预测结果，跳过")
                continue
            
            pred_fp = pred_dict[spectrum_id]
            
            # 鉴定化合物
            candidates = identify_compound(spectrum, pred_fp, pubchem_db, mass_tolerance, max_candidates)
            
            # 提取正确的InChIKey和候选InChIKey
            true_inchikey = spectrum['inchikey']
            
            # 从参考库中获取正确化合物的信息
            true_compound_info = reference_library.get(true_inchikey, {})
            
            # 如果InChIKey在参考库中不存在，可以添加警告日志
            if not true_compound_info:
                print(f"警告: InChIKey {true_inchikey} 在参考库中未找到")
            
            candidate_inchikeys = [c['inchikey'] for c in candidates]
            
            # 检查Top-N结果中是否包含正确答案
            identification_result = {
                'spectrum_id': spectrum_id,
                'true_inchikey': true_inchikey,
                'neutral_mass': spectrum['neutral_mass'],
                'precursor_mz': spectrum['precursor_mz'],
                'candidate_count': len(candidates),
                'top_candidates': [],
                'true_compound_info': true_compound_info
            }
            
            correct_rank = None
            for n in top_n_list:
                # 检查正确InChIKey是否在前N个候选中
                if n <= len(candidate_inchikeys) and true_inchikey in candidate_inchikeys[:n]:
                    results['correct'][n] += 1
                    identification_result[f'correct_in_top_{n}'] = True
                    
                    # 找出正确答案的排名
                    if correct_rank is None:
                        correct_rank = candidate_inchikeys.index(true_inchikey) + 1
                        identification_result['correct_rank'] = correct_rank
                else:
                    identification_result[f'correct_in_top_{n}'] = False
            
            if correct_rank is None and true_inchikey in candidate_inchikeys:
                correct_rank = candidate_inchikeys.index(true_inchikey) + 1
                identification_result['correct_rank'] = correct_rank
            
            # 记录前5个候选
            for j, candidate in enumerate(candidates[:5]):
                # 检查候选化合物是否在参考库中
                candidate_info = reference_library.get(candidate['inchikey'], {})
                
                candidate_entry = {
                    'rank': j + 1,
                    'cid': candidate['cid'],
                    'inchikey': candidate['inchikey'],
                    'similarity': candidate['similarity'],
                    'mass_error': candidate['mass_error'],
                    'is_correct': candidate['inchikey'] == true_inchikey,
                    'reference_info': candidate_info
                }
                
                identification_result['top_candidates'].append(candidate_entry)
            
            # 添加到结果列表
            results['details'].append(identification_result)
            
            # 定期保存中间结果，防止程序崩溃导致所有进度丢失
            if len(results['details']) % 50 == 0:
                interim_file = os.path.join(output_dir, f"interim_results_{len(results['details'])}.json")
                try:
                    with open(interim_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f)
                    # 只保留最新的5个中间结果文件
                    interim_files = [f for f in os.listdir(output_dir) if f.startswith("interim_results_")]
                    if len(interim_files) > 5:
                        oldest_file = min(interim_files, key=lambda x: os.path.getmtime(os.path.join(output_dir, x)))
                        os.remove(os.path.join(output_dir, oldest_file))
                except Exception as e:
                    print(f"保存中间结果时出错: {str(e)}")
            
            # 释放内存
            candidates = None
        
        # 每个批次结束后强制垃圾回收
        import gc
        gc.collect()
    
    # 计算每个Top-N的准确率
    for n in top_n_list:
        accuracy = results['correct'][n] / results['total'] if results['total'] > 0 else 0
        results[f'accuracy_top_{n}'] = accuracy
        print(f"Top-{n} 准确率: {accuracy:.4f} ({results['correct'][n]}/{results['total']})")
    
    # 生成详细的报告，包含参考库信息
    generate_detailed_report(results, reference_library, output_dir)
    
    # 保存详细结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"identification_results_{timestamp}.json")
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # 保存摘要结果
    summary_file = os.path.join(output_dir, f"summary_{timestamp}.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"质谱鉴定评估结果\n")
        f.write(f"模型: {args.model_path}\n")
        f.write(f"日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总谱图数: {results['total']}\n")
        f.write(f"质量容差: {mass_tolerance} Da\n\n")
        
        f.write("准确率统计:\n")
        for n in top_n_list:
            accuracy = results[f'accuracy_top_{n}']
            f.write(f"Top-{n}: {accuracy:.4f} ({results['correct'][n]}/{results['total']})\n")
    
    # 删除进度文件
    try:
        os.remove(os.path.join(output_dir, "progress.txt"))
    except:
        pass
    
    print(f"评估完成，结果已保存至 {output_dir}")
    return results

def analyze_results_with_reference(results, reference_library, output_dir):
    """使用参考库分析鉴定结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_file = os.path.join(output_dir, f"analysis_{timestamp}.txt")
    
    # 按来源统计成功率
    source_stats = defaultdict(lambda: {'total': 0, 'correct_top1': 0, 'correct_top5': 0})
    formula_stats = defaultdict(lambda: {'total': 0, 'correct_top1': 0, 'correct_top5': 0})
    
    # 分子质量范围统计
    mass_ranges = {
        '小分子 (<200 Da)': {'total': 0, 'correct_top1': 0, 'correct_top5': 0},
        '中分子 (200-500 Da)': {'total': 0, 'correct_top1': 0, 'correct_top5': 0},
        '大分子 (>500 Da)': {'total': 0, 'correct_top1': 0, 'correct_top5': 0}
    }
    
    # 分析错误案例的特点
    errors_by_mass = []
    errors_by_similarity = []
    
    for detail in results['details']:
        true_inchikey = detail['true_inchikey']
        true_info = reference_library.get(true_inchikey, {})
        
        # 获取化合物信息
        source = true_info.get('source', 'Unknown')
        formula = true_info.get('formula', 'Unknown')
        mol_weight = true_info.get('molecular_weight', 0)
        
        # 统计按来源
        source_stats[source]['total'] += 1
        formula_stats[formula]['total'] += 1
        
        # 统计按质量范围
        if mol_weight < 200:
            mass_range = '小分子 (<200 Da)'
        elif mol_weight <= 500:
            mass_range = '中分子 (200-500 Da)'
        else:
            mass_range = '大分子 (>500 Da)'
        
        mass_ranges[mass_range]['total'] += 1
        
        # 统计Top-1和Top-5准确率
        if detail.get('correct_in_top_1', False):
            source_stats[source]['correct_top1'] += 1
            formula_stats[formula]['correct_top1'] += 1
            mass_ranges[mass_range]['correct_top1'] += 1
        
        if detail.get('correct_in_top_5', False):
            source_stats[source]['correct_top5'] += 1
            formula_stats[formula]['correct_top5'] += 1
            mass_ranges[mass_range]['correct_top5'] += 1
        
        # 分析错误案例 (不在Top-5中的)
        if not detail.get('correct_in_top_5', False):
            # 如果有顶部候选
            if detail['top_candidates']:
                # 获取顶部候选的相似度
                top_similarity = detail['top_candidates'][0]['similarity']
                
                # 如果正确答案在候选中，获取其排名和相似度
                if 'correct_rank' in detail:
                    correct_rank = detail['correct_rank']
                    # 尝试找到正确答案的相似度
                    correct_similarity = None
                    for cand in detail['top_candidates']:
                        if cand['is_correct']:
                            correct_similarity = cand['similarity']
                            break
                    
                    # 如果找到了正确答案的相似度，记录错误信息
                    if correct_similarity is not None:
                        error_info = {
                            'spectrum_id': detail['spectrum_id'],
                            'true_inchikey': true_inchikey,
                            'mol_weight': mol_weight,
                            'correct_rank': correct_rank,
                            'top_similarity': top_similarity,
                            'correct_similarity': correct_similarity,
                            'similarity_diff': top_similarity - correct_similarity,
                            'formula': formula,
                            'source': source
                        }
                        
                        errors_by_similarity.append(error_info)
                
                # 记录质量相关错误
                if 'neutral_mass' in detail:
                    # 计算质量误差
                    neutral_mass = detail['neutral_mass']
                    mass_error = abs(neutral_mass - mol_weight) if mol_weight else None
                    
                    if mass_error is not None:
                        error_info = {
                            'spectrum_id': detail['spectrum_id'],
                            'true_inchikey': true_inchikey,
                            'mol_weight': mol_weight,
                            'calculated_mass': neutral_mass,
                            'mass_error': mass_error,
                            'mass_error_ppm': (mass_error / mol_weight) * 1e6 if mol_weight else None,
                            'formula': formula,
                            'source': source
                        }
                        
                        errors_by_mass.append(error_info)
    
    # 开始写入分析文件
    with open(analysis_file, 'w', encoding='utf-8') as f:
        f.write("=== 质谱鉴定结果深度分析 ===\n")
        f.write(f"日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总谱图数: {results['total']}\n\n")
        
        # 写入总体准确率
        f.write("总体准确率:\n")
        for n in sorted([int(k.split('_')[-1]) for k in results.keys() if k.startswith('accuracy_top_')]):
            accuracy = results[f'accuracy_top_{n}']
            f.write(f"- Top-{n}: {accuracy:.4f} ({results['correct'][n]}/{results['total']})\n")
        f.write("\n")
        
        # 按来源统计
        f.write("=== 按化合物来源分析 ===\n\n")
        for source, stats in sorted(source_stats.items(), key=lambda x: x[1]['total'], reverse=True):
            top1_acc = stats['correct_top1'] / stats['total'] if stats['total'] > 0 else 0
            top5_acc = stats['correct_top5'] / stats['total'] if stats['total'] > 0 else 0
            f.write(f"来源: {source}\n")
            f.write(f"- 样本数: {stats['total']}\n")
            f.write(f"- Top-1准确率: {top1_acc:.4f} ({stats['correct_top1']}/{stats['total']})\n")
            f.write(f"- Top-5准确率: {top5_acc:.4f} ({stats['correct_top5']}/{stats['total']})\n\n")
        
        # 按分子量范围统计
        f.write("=== 按分子量范围分析 ===\n\n")
        for mass_range, stats in sorted(mass_ranges.items(), key=lambda x: x[0]):
            if stats['total'] == 0:
                continue
            top1_acc = stats['correct_top1'] / stats['total']
            top5_acc = stats['correct_top5'] / stats['total']
            f.write(f"范围: {mass_range}\n")
            f.write(f"- 样本数: {stats['total']}\n")
            f.write(f"- Top-1准确率: {top1_acc:.4f} ({stats['correct_top1']}/{stats['total']})\n")
            f.write(f"- Top-5准确率: {top5_acc:.4f} ({stats['correct_top5']}/{stats['total']})\n\n")
        
        # 分析分子式与成功率的关系
        f.write("=== 按分子式复杂度分析 ===\n\n")
        # 使用分子式中的原子数作为复杂度简易指标
        formula_complexity = {}
        for formula, stats in formula_stats.items():
            if stats['total'] < 5:  # 忽略样本太少的
                continue
            # 简单提取碳原子数(如果有的话)
            carbon_count = 0
            if 'C' in formula:
                parts = formula.split('C')
                if len(parts) > 1:
                    carbon_str = ""
                    for c in parts[1]:
                        if c.isdigit():
                            carbon_str += c
                        else:
                            break
                    if carbon_str:
                        carbon_count = int(carbon_str)
                    else:
                        carbon_count = 1
            
            formula_complexity[formula] = {
                'carbon_count': carbon_count,
                'stats': stats
            }
        
        # 按碳原子数分组
        carbon_groups = {
            '低碳 (C<10)': {'total': 0, 'correct_top1': 0, 'correct_top5': 0},
            '中碳 (C10-20)': {'total': 0, 'correct_top1': 0, 'correct_top5': 0},
            '高碳 (C>20)': {'total': 0, 'correct_top1': 0, 'correct_top5': 0}
        }
        
        for formula, data in formula_complexity.items():
            carbon_count = data['carbon_count']
            stats = data['stats']
            
            if carbon_count < 10:
                group = '低碳 (C<10)'
            elif carbon_count <= 20:
                group = '中碳 (C10-20)'
            else:
                group = '高碳 (C>20)'
            
            carbon_groups[group]['total'] += stats['total']
            carbon_groups[group]['correct_top1'] += stats['correct_top1']
            carbon_groups[group]['correct_top5'] += stats['correct_top5']
        
        for group, stats in sorted(carbon_groups.items(), key=lambda x: x[0]):
            if stats['total'] == 0:
                continue
            top1_acc = stats['correct_top1'] / stats['total']
            top5_acc = stats['correct_top5'] / stats['total']
            f.write(f"碳原子数: {group}\n")
            f.write(f"- 样本数: {stats['total']}\n")
            f.write(f"- Top-1准确率: {top1_acc:.4f} ({stats['correct_top1']}/{stats['total']})\n")
            f.write(f"- Top-5准确率: {top5_acc:.4f} ({stats['correct_top5']}/{stats['total']})\n\n")
        
        # 错误案例分析
        f.write("=== 错误案例分析 ===\n\n")
        # 按相似度差异排序的错误案例
        if errors_by_similarity:
            f.write("相似度分析 (Top-5外的前10个最接近案例):\n")
            errors_by_similarity.sort(key=lambda x: x['similarity_diff'])
            for i, error in enumerate(errors_by_similarity[:10]):
                f.write(f"{i+1}. 谱图ID: {error['spectrum_id']}\n")
                f.write(f"   - 真实InChIKey: {error['true_inchikey']}\n")
                f.write(f"   - 分子量: {error['mol_weight']}\n")
                f.write(f"   - 正确答案排名: {error['correct_rank']}\n")
                f.write(f"   - 顶部候选相似度: {error['top_similarity']:.4f}\n")
                f.write(f"   - 正确答案相似度: {error['correct_similarity']:.4f}\n")
                f.write(f"   - 相似度差异: {error['similarity_diff']:.4f}\n")
                f.write(f"   - 来源: {error['source']}\n\n")
        
        # 按质量误差排序的错误案例
        if errors_by_mass:
            f.write("质量误差分析 (Top-5外的前10个最小质量误差案例):\n")
            errors_by_mass.sort(key=lambda x: x['mass_error'])
            for i, error in enumerate(errors_by_mass[:10]):
                f.write(f"{i+1}. 谱图ID: {error['spectrum_id']}\n")
                f.write(f"   - 真实InChIKey: {error['true_inchikey']}\n")
                f.write(f"   - 真实分子量: {error['mol_weight']}\n")
                f.write(f"   - 计算分子量: {error['calculated_mass']}\n")
                f.write(f"   - 质量误差: {error['mass_error']:.6f} Da\n")
                if error['mass_error_ppm'] is not None:
                    f.write(f"   - 质量误差: {error['mass_error_ppm']:.2f} ppm\n")
                f.write(f"   - 来源: {error['source']}\n\n")
        # 总结
        f.write("=== 结论与建议 ===\n\n")
        
        # 找出表现最好和最差的样本类别
        best_source = max(source_stats.items(), key=lambda x: x[1]['correct_top1']/x[1]['total'] if x[1]['total'] > 5 else 0)
        worst_source = min(source_stats.items(), key=lambda x: x[1]['correct_top1']/x[1]['total'] if x[1]['total'] > 5 else float('inf'))
        
        best_mass_range = max(mass_ranges.items(), key=lambda x: x[1]['correct_top1']/x[1]['total'] if x[1]['total'] > 0 else 0)
        worst_mass_range = min(mass_ranges.items(), key=lambda x: x[1]['correct_top1']/x[1]['total'] if x[1]['total'] > 0 else float('inf'))
        
        f.write("1. 表现分析:\n")
        
        if best_source[1]['total'] > 5:
            best_source_acc = best_source[1]['correct_top1'] / best_source[1]['total']
            f.write(f"   - 表现最好的化合物来源: {best_source[0]}, Top-1准确率: {best_source_acc:.4f}\n")
        
        if worst_source[1]['total'] > 5:
            worst_source_acc = worst_source[1]['correct_top1'] / worst_source[1]['total']
            f.write(f"   - 表现最差的化合物来源: {worst_source[0]}, Top-1准确率: {worst_source_acc:.4f}\n")
        
        if best_mass_range[1]['total'] > 0:
            best_mass_acc = best_mass_range[1]['correct_top1'] / best_mass_range[1]['total']
            f.write(f"   - 表现最好的分子量范围: {best_mass_range[0]}, Top-1准确率: {best_mass_acc:.4f}\n")
        
        if worst_mass_range[1]['total'] > 0:
            worst_mass_acc = worst_mass_range[1]['correct_top1'] / worst_mass_range[1]['total']
            f.write(f"   - 表现最差的分子量范围: {worst_mass_range[0]}, Top-1准确率: {worst_mass_acc:.4f}\n")
        
        f.write("\n2. 可能的改进方向:\n")
        
        # 基于错误案例分析提出建议
        if errors_by_similarity:
            avg_similarity_diff = sum(e['similarity_diff'] for e in errors_by_similarity) / len(errors_by_similarity)
            f.write(f"   - 相似度差异分析: 平均相似度差异为 {avg_similarity_diff:.4f}\n")
            
            if avg_similarity_diff < 0.1:
                f.write("   - 建议: 考虑优化模型以提高区分能力，相似结构的区分度较低\n")
            else:
                f.write("   - 建议: 可能需要增加更多相似结构的训练样本，以提高特异性\n")
        
        if errors_by_mass:
            avg_mass_error_ppm = np.mean([e['mass_error_ppm'] for e in errors_by_mass if e['mass_error_ppm'] is not None])
            f.write(f"   - 质量误差分析: 平均质量误差为 {avg_mass_error_ppm:.2f} ppm\n")
            
            if avg_mass_error_ppm > 10:
                f.write("   - 建议: 考虑提高质量精度，或优化质量容差设置\n")
        
        # 查看Top-1和Top-5的准确率差异
        top1_acc = results.get('accuracy_top_1', 0)
        top5_acc = results.get('accuracy_top_5', 0)
        diff = top5_acc - top1_acc
        
        if diff > 0.2:
            f.write("   - Top-1和Top-5准确率相差较大，说明正确答案经常在候选列表中但排名不高\n")
            f.write("   - 建议: 优化相似度计算方法，或考虑组合多种指纹类型改进排序\n")
        
        # 为特定类别的化合物提供建议
        if worst_source_acc < 0.5 and worst_source[1]['total'] > 10:
            f.write(f"   - 针对{worst_source[0]}类化合物的建议: 考虑增加此类样本的训练数据或专门优化\n")
        
        if worst_mass_acc < 0.5 and worst_mass_range[1]['total'] > 10:
            f.write(f"   - 针对{worst_mass_range[0]}的建议: 可能需要针对此质量范围的化合物进行特殊处理\n")
        
        f.write("\n3. 总结:\n")
        overall_acc = results.get('accuracy_top_1', 0)
        if overall_acc >= 0.8:
            f.write("   - 模型整体表现良好，Top-1准确率达到较高水平\n")
        elif overall_acc >= 0.5:
            f.write("   - 模型表现一般，有进一步优化的空间\n")
        else:
            f.write("   - 模型表现不佳，需要重新考虑算法或增加训练数据\n")
        
        f.write(f"   - 鉴定日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"   - 使用者: snow0726\n")
    
    print(f"分析报告已保存至: {analysis_file}")
    
    # 返回一些关键统计数据供进一步使用
    return {
        'source_stats': source_stats,
        'mass_ranges': mass_ranges,
        'error_analysis': {
            'similarity_errors': errors_by_similarity,
            'mass_errors': errors_by_mass
        }
    }

def main():
    """主函数"""
    global args
    args = parse_args()
    
    # 解析Top-N列表
    top_n_list = [int(n) for n in args.top_n.split(',')]
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 开始计时
    start_time = time.time()
    
    # 加载预训练模型
    model, device = load_model(args.model_path)
    
    # 加载鉴定数据集
    spectra_data = load_identification_dataset(args.ident_dataset)
    
    # 加载参考库
    reference_library = load_reference_library(args.reference_library)
    
    # 创建数据集对象
    ident_dataset = IdentificationDataset(spectra_data)
    print(f"创建了鉴定数据集，包含 {len(ident_dataset)} 条记录")
    
    # 使用模型预测分子指纹
    pred_results = predict_fingerprints(model, device, ident_dataset, batch_size=args.batch_size)
    
    # 加载PubChem数据库
    pubchem_db = PubChemDatabase(
        args.cid_inchikey,
        args.cid_mass,
        args.cid_smiles
    )
    
    # 如果指定了缓存指纹
    if args.cache_fingerprints:
        if os.path.exists(args.cache_file):
            try:
                print(f"加载PubChem指纹缓存: {args.cache_file}")
                with open(args.cache_file, 'rb') as f:
                    pubchem_db.fingerprints_cache = pickle.load(f)
                print(f"加载了 {len(pubchem_db.fingerprints_cache)} 条指纹缓存")
            except Exception as e:
                print(f"加载指纹缓存失败: {str(e)}")
    
    # 执行鉴定评估
    results = evaluate_identification(
        spectra_data,
        pred_results,
        pubchem_db,
        reference_library,  # 确保传递参考库
        top_n_list=top_n_list,
        mass_tolerance=args.mass_tolerance,
        max_candidates=args.max_candidates,
        output_dir=args.output_dir,
        batch_size=args.eval_batch_size
    )
    analyze_results_with_reference(results, reference_library, args.output_dir)
    # 如果指定了缓存指纹
    if args.cache_fingerprints and pubchem_db.fingerprints_cache:
        try:
            print(f"保存PubChem指纹缓存: {args.cache_file}")
            with open(args.cache_file, 'wb') as f:
                pickle.dump(pubchem_db.fingerprints_cache, f)
            print(f"保存了 {len(pubchem_db.fingerprints_cache)} 条指纹缓存")
        except Exception as e:
            print(f"保存指纹缓存失败: {str(e)}")
    
    # 计算总耗时
    elapsed_time = time.time() - start_time
    print(f"总耗时: {elapsed_time:.2f} 秒")
    
    # 输出准确率摘要
    print("\n鉴定准确率摘要:")
    for n in top_n_list:
        accuracy = results[f'accuracy_top_{n}']
        print(f"Top-{n}: {accuracy:.4f} ({results['correct'][n]}/{results['total']})")
    
    return results

def analyze_rank_distribution(results):
    """分析正确候选的排名分布"""
    correct_ranks = []
    for detail in results['details']:
        if 'correct_rank' in detail:
            correct_ranks.append(detail['correct_rank'])
    
    if not correct_ranks:
        return None
    
    # 计算排名统计
    rank_stats = {
        'mean': np.mean(correct_ranks),
        'median': np.median(correct_ranks),
        'min': np.min(correct_ranks),
        'max': np.max(correct_ranks),
        'std': np.std(correct_ranks),
        'count': len(correct_ranks),
        'distribution': {}
    }
    
    # 计算排名分布
    for rank in sorted(set(correct_ranks)):
        rank_stats['distribution'][rank] = correct_ranks.count(rank)
    
    return rank_stats

def analyze_success_failure_cases(results, pubchem_db, reference_library):
    """分析成功和失败的案例"""
    analysis = {
        'success': [],
        'failure': []
    }
    
    # 分析Top-5中的成功和失败案例
    for detail in results['details']:
        if detail.get('correct_in_top_5', False):
            # 成功案例
            inchikey = detail['true_inchikey']
            ref = reference_library.get(inchikey, {})
            
            success_info = {
                'spectrum_id': detail['spectrum_id'],
                'inchikey': inchikey,
                'name': ref.get('name', 'Unknown'),
                'formula': ref.get('formula', 'Unknown'),
                'correct_rank': detail.get('correct_rank', None),
                'candidate_count': detail['candidate_count'],
                'top_candidate_similarity': detail['top_candidates'][0]['similarity'] if detail['top_candidates'] else None,
                'correct_candidate_similarity': next((c['similarity'] for c in detail['top_candidates'] if c['is_correct']), None)
            }
            
            analysis['success'].append(success_info)
        else:
            # 失败案例
            inchikey = detail['true_inchikey']
            ref = reference_library.get(inchikey, {})
            
            failure_info = {
                'spectrum_id': detail['spectrum_id'],
                'inchikey': inchikey,
                'name': ref.get('name', 'Unknown'),
                'formula': ref.get('formula', 'Unknown'),
                'correct_rank': detail.get('correct_rank', None),
                'candidate_count': detail['candidate_count'],
                'top_candidates': detail['top_candidates']
            }
            
            analysis['failure'].append(failure_info)
    
    return analysis

if __name__ == "__main__":
    """
    示例用法：
    
    python identification_models.py \
        --model_path /data/zhangxiaofeng/code/code/spectomol/runs/model_featurefusion_hunhe_nist_hdim_2048_layers_6_bs_512_adam/489_best_checkpoint.pt \
        --model_type featurefusion \
        --ident_dataset /data/zhangxiaofeng/code/code/data/Identification_data/identification_dataset.tsv \
        --reference_library /data/zhangxiaofeng/code/code/data/Identification_data/reference_library.tsv \
        --cid_inchikey /data/zhangxiaofeng/code/code/data/PubChem/CID-InChI-Key.txt \
        --cid_mass /data/zhangxiaofeng/code/code/data/PubChem/CID-Mass.txt \
        --cid_smiles /data/zhangxiaofeng/code/code/data/PubChem/CID-SMILES.txt \
        --mass_tolerance 0.01 \
        --top_n 1,3,5,10,20 \
        --output_dir /data/zhangxiaofeng/code/code/spectomol/identification_results/featurefusion \
        --cache_fingerprints \
        --cache_file /data/zhangxiaofeng/code/code/data/PubChem/pubchem_fingerprints.pkl
    """
    try:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"开始运行时间: {current_time}")
        print(f"用户: snow0726")
        
        # 运行主函数
        results = main()
        
        # 分析排名分布
        rank_stats = analyze_rank_distribution(results)
        if rank_stats:
            print("\n正确候选排名分布:")
            print(f"- 平均排名: {rank_stats['mean']:.2f}")
            print(f"- 中位数排名: {rank_stats['median']:.1f}")
            print(f"- 最小排名: {rank_stats['min']}")
            print(f"- 最大排名: {rank_stats['max']}")
            print(f"- 标准差: {rank_stats['std']:.2f}")
            print(f"- 总计: {rank_stats['count']} 条记录有正确候选")
        
        print("\n鉴定完成!")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    