from typing import Any
import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as gnn
from rdkit import Chem
from rdkit.Chem import AllChem,MACCSkeys
from molxspec import egnn
import numpy as np
from rdkit.Chem import DataStructs
from training_setup import setup_device
class ResBlock(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, inputs: Any) -> torch.Tensor:
        return self.module(inputs) + inputs



class NeimsBlock(nn.Module):
    """from the NEIMS paper (uses LeakyReLU instead of ReLU)"""
    def __init__(self, in_dim, out_dim, dropout):

        super(NeimsBlock, self).__init__()
        bottleneck_factor = 0.5
        bottleneck_size = int(round(bottleneck_factor * out_dim))
        self.in_batch_norm = nn.BatchNorm1d(in_dim)
        self.in_activation = nn.LeakyReLU()
        self.in_linear = nn.Linear(in_dim, bottleneck_size)
        self.out_batch_norm = nn.BatchNorm1d(bottleneck_size)
        self.out_linear = nn.Linear(bottleneck_size, out_dim)
        self.out_activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):

        h = x
        h = self.in_batch_norm(h)
        h = self.in_activation(h)
        h = self.dropout(h)
        h = self.in_linear(h)
        h = self.out_batch_norm(h)
        h = self.out_activation(h)
        h = self.out_linear(h)
        return h

class Mol2SpecSimple(nn.Module):
    def __init__(self, molecule_dim: int, prop_dim: int, hdim: int, n_layers: int):
        super().__init__()
        self.kwargs = dict(
            molecule_dim=molecule_dim,
            prop_dim=prop_dim,
            hdim=hdim,
            n_layers=n_layers
        )

        self.meta = nn.Parameter(torch.empty(0))
        self.molecule_dim = molecule_dim
        self.prop_dim = prop_dim
        dropout_p = 0.1
        res_blocks = [
            ResBlock(
                nn.Sequential(
                    nn.Dropout(p=dropout_p),
                    nn.Linear(molecule_dim, hdim),
                    nn.BatchNorm1d(hdim),
                    nn.SiLU(),
                    nn.Linear(hdim, molecule_dim),
                )
            )
            for _ in range(n_layers)
        ]
        self.mlp_layers = nn.Sequential(
            *res_blocks,
            nn.BatchNorm1d(molecule_dim),
            nn.Linear(molecule_dim, prop_dim),
            # nn.Sigmod()
            nn.SiLU()
        )
    def forward(self, mol_vec: torch.Tensor) -> torch.Tensor:
        mol_vec = mol_vec.type(torch.FloatTensor).to(self.meta.device)
        mz_res = self.mlp_layers(mol_vec)
        return mz_res


class My_model2(nn.Module):
    def __init__(self, molecule_dim: int, prop_dim: int, hdim: int, n_layers: int):
        super().__init__()
        self.kwargs = dict(
            molecule_dim=molecule_dim,
            prop_dim=prop_dim,
            hdim=hdim,
            n_layers=n_layers
        )
        self.sigmod=nn.Sigmoid()
        # self.NeimsBlock=NeimsBlock(molecule_dim,prop_dim,0.1)
        self.NeimsBlock = NeimsBlock(4000, 1038, 0.1)
        self.meta = nn.Parameter(torch.empty(0))
    def forward(self, mol_vec: torch.Tensor) -> torch.Tensor:
        mol_vec = mol_vec.type(torch.FloatTensor).to(self.meta.device)
        # print("Input shape:", mol_vec.shape)  # 添加这一行进行调试
        mz_res=self.NeimsBlock(mol_vec)
        # print("NeimsBlock output shape:", mz_res.shape)  # 添加这一行进行调试
        return self.sigmod(mz_res)




class My_model1(nn.Module):
    def __init__(self, molecule_dim: int, prop_dim: int, hdim: int, n_layers: int):
        super().__init__()
        self.kwargs = dict(
            molecule_dim=molecule_dim,
            prop_dim=prop_dim,
            hdim=hdim,
            n_layers=n_layers
        )
        self.NeimsBlock=NeimsBlock(4000,4000,0.1)
        self.meta = nn.Parameter(torch.empty(0))
        self.molecule_dim = molecule_dim
        self.prop_dim = prop_dim
        dropout_p = 0.1
        res_blocks = [
            ResBlock(
                nn.Sequential(
                    nn.Dropout(p=dropout_p),
                    nn.Linear(molecule_dim, hdim),
                    nn.SiLU(),
                    nn.Linear(hdim, molecule_dim),
                )
            )
            for _ in range(n_layers)
        ]
        self.mlp_layers = nn.Sequential(
            *res_blocks,
            nn.Linear(molecule_dim, prop_dim),
            # nn.Sigmod()
        )
    def forward(self, mol_vec: torch.Tensor) -> torch.Tensor:
        mol_vec = mol_vec.type(torch.FloatTensor).to(self.meta.device)
        mz_res = self.mlp_layers(mol_vec)
        mz_res=self.NeimsBlock(mz_res)
        return mz_res


class SelfAttentionBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(SelfAttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads)
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x, _ = self.attention(x, x, x)
        x += residual  
        x = self.fc(x)
        x = self.relu(x)
        return x

class MyEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, output_dim):
        super(MyEncoder, self).__init__()

        self.kwargs = dict(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            output_dim=output_dim
        )
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.self_attention_blocks = nn.ModuleList([
            SelfAttentionBlock(hidden_dim, hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x=x.to(torch.float)
        x = self.fc(x)
        for self_attention_block in self.self_attention_blocks:
            x = self_attention_block(x)
        x = self.fc_out(x)
        return x

class My_Mol2SpecSimple(nn.Module):
    def __init__(self, molecule_dim, prop_dim, hdim, n_layers, input_dim):
        super().__init__()
        self.kwargs = dict(
            molecule_dim=molecule_dim,
            prop_dim=prop_dim,
            hdim=hdim,
            n_layers=n_layers,
            input_dim=input_dim,
        )
        self.MyEncoder=MyEncoder(input_dim,hdim,3,8,molecule_dim)
        dropout_p = 0.1
        res_blocks = [
            ResBlock(
                nn.Sequential(
                    nn.Dropout(p=dropout_p),
                    nn.Linear(molecule_dim, hdim),
                    nn.SiLU(),
                    nn.Linear(hdim, molecule_dim),
                )
            )
            for _ in range(n_layers)
        ]
        self.mlp_layers = nn.Sequential(
            *res_blocks,
            nn.Linear(molecule_dim, prop_dim),
            # nn.Sigmod()
        )

    def forward(self, x1):
        out=self.MyEncoder(x1)
        final_output = self.mlp_layers(out)
        return final_output




class SelfAttentionBlock1(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout_rate=0.1):
        super(SelfAttentionBlock1, self).__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout_rate)
        self.norm1 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim)
        )
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Self-attention with residual connection and layer normalization
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.attention(x, x, x)
        x = residual + self.dropout1(attn_output)
        
        # Feed-forward with residual connection and layer normalization
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout2(self.fc(x))
        return x

class MyEncoder1(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, output_dim, dropout_rate=0.1):
        super(MyEncoder1, self).__init__()

        self.kwargs = dict(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            output_dim=output_dim
        )
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        self.self_attention_blocks = nn.ModuleList([
            SelfAttentionBlock1(hidden_dim, hidden_dim*4, num_heads, dropout_rate)
            for _ in range(num_layers)
        ])
        
        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = x.to(torch.float)
        
        # Reshape input to sequence format for transformer
        batch_size = x.size(0)
        x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        # Project to hidden dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.positional_encoding
        
        # Apply transformer blocks
        for self_attention_block in self.self_attention_blocks:
            x = self_attention_block(x)
        
        # Get the output
        x = self.output_projection(x).squeeze(1)
        return x

class FeatureInteraction(nn.Module):
    """特征交互模块,使MLP和Transformer特征相互增强"""
    def __init__(self, feature_dim, hidden_dim, dropout_rate=0.2):
        super(FeatureInteraction, self).__init__()
        self.mlp_enhance = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        self.transformer_enhance = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
    def forward(self, mlp_features, transformer_features):
        # 特征融合
        concat_features = torch.cat([mlp_features, transformer_features], dim=1)
        
        # 增强MLP特征
        enhanced_mlp = self.mlp_enhance(concat_features) + mlp_features  # 残差连接
        
        # 增强Transformer特征
        enhanced_transformer = self.transformer_enhance(concat_features) + transformer_features  # 残差连接
        
        return enhanced_mlp, enhanced_transformer

class AttentiveFeatureFusion(nn.Module):
    """使用注意力机制动态融合特征"""
    def __init__(self, feature_dim, hidden_dim, dropout_rate=0.2):
        super(AttentiveFeatureFusion, self).__init__()
        
        # 特征重要性权重计算
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1)
        )
        
        # 融合后特征转换
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, mlp_features, transformer_features):
        # 计算特征重要性权重
        concat_features = torch.cat([mlp_features, transformer_features], dim=1)
        weights = self.attention(concat_features)
        
        # 加权融合
        weighted_mlp = mlp_features * weights[:, 0:1]
        weighted_transformer = transformer_features * weights[:, 1:2]
        fused_features = weighted_mlp + weighted_transformer
        
        # 融合后转换
        output = self.fusion(fused_features) + fused_features  # 残差连接
        
        return output, {
            'mlp_weight': weights[:, 0].mean().item(),
            'transformer_weight': weights[:, 1].mean().item()
        }

class Spec2MolFeaturefusion(nn.Module):
    def __init__(self, molecule_dim: int, prop_dim: int, hdim: int, n_layers: int):
        super(Spec2MolFeaturefusion, self).__init__()
        self.kwargs = dict(
            molecule_dim=molecule_dim,            
            prop_dim=prop_dim,
            hdim=hdim,
            n_layers=n_layers
        )
        self.meta = nn.Parameter(torch.empty(0))
        self.molecule_dim = molecule_dim
        self.prop_dim = prop_dim
        
        # 子模型参数设置
        transformer_hidden_dim = 2048  # Transformer隐藏层维度
        # transformer_hidden_dim = 1024  # Transformer隐藏层维度
        transformer_layers = 4  # Transformer层数
        # transformer_layers = 3  # Transformer层数
        transformer_heads = 8  # Transformer头数
        dropout_rate = 0.2  # 正则化系数
        
        # 基础特征提取器
        self.mlp = Mol2SpecSimple(
            molecule_dim=molecule_dim, 
            prop_dim=prop_dim,
            hdim=hdim, 
            n_layers=n_layers
        )
        
        self.transformer = MyEncoder1(
            input_dim=molecule_dim, 
            hidden_dim=transformer_hidden_dim, 
            num_layers=transformer_layers, 
            num_heads=transformer_heads, 
            output_dim=prop_dim,
            dropout_rate=dropout_rate
            # dropout_rate=0.3
        )
        
        # 特征交互模块
        self.feature_interaction = FeatureInteraction(
            feature_dim=prop_dim,
            hidden_dim=hdim,
            # hidden_dim=2048,
            dropout_rate=dropout_rate
        )
        
        # 特征融合模块
        self.feature_fusion = AttentiveFeatureFusion(
            feature_dim=prop_dim,
            hidden_dim=hdim,
            dropout_rate=dropout_rate
        )
        
        # 残差连接投影（将原始MLP特征投影到最终特征空间）
        self.residual_projection = nn.Linear(prop_dim, prop_dim)
        
        # 最终输出层
        self.output_layers = nn.Sequential(
            nn.Linear(prop_dim, hdim),
            nn.LayerNorm(hdim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hdim, prop_dim),
            nn.SiLU()
        )
        
        # # 损失函数权重（可学习）
        # self.loss_weights = nn.Parameter(torch.ones(3))  # [mlp, transformer, fusion]

    def forward(self, mol_vec: torch.Tensor) -> torch.Tensor:
        device = self.meta.device
        mol_vec = mol_vec.type(torch.FloatTensor).to(device)
        
        # 1. 基础特征提取
        mlp_features = self.mlp(mol_vec)
        transformer_features = self.transformer(mol_vec)
        
        # 2. 特征交互
        enhanced_mlp, enhanced_transformer = self.feature_interaction(mlp_features, transformer_features)
        
        # 3. 特征融合
        fused_features, attention_stats = self.feature_fusion(enhanced_mlp, enhanced_transformer)
        
        # 4. 残差连接（从原始MLP特征）
        residual = self.residual_projection(mlp_features)
        fused_features = fused_features + residual
        
        # 5. 最终输出
        output = self.output_layers(fused_features)
        
        return output


# new

class EnhancedFeatureFusion(nn.Module):
    def __init__(self, feature_dim, hidden_dims=[1024, 512, 256], dropout_rate=0.2):
        super(EnhancedFeatureFusion, self).__init__()
        self.original_feature_dim = feature_dim
        
        # 计算内部使用的可被8整除的维度
        self.internal_dim = ((feature_dim + 5) // 6) * 6
        print(f"特征融合：原始维度 {feature_dim}，内部处理维度 {self.internal_dim}")
        
        # 添加投影层
        self.input_projection_mlp = nn.Linear(feature_dim, self.internal_dim)
        self.input_projection_transformer = nn.Linear(feature_dim, self.internal_dim)
        self.output_projection = nn.Linear(self.internal_dim, feature_dim)
        
        # 1. 多层特征变换
        self.mlp_transforms = nn.ModuleList()
        self.transformer_transforms = nn.ModuleList()
        
        for hidden_dim in hidden_dims:
            # MLP特征变换 - 使用internal_dim
            mlp_transform = nn.Sequential(
                nn.Linear(self.internal_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, self.internal_dim),
                nn.LayerNorm(self.internal_dim)
            )
            self.mlp_transforms.append(mlp_transform)
            
            # Transformer特征变换 - 使用internal_dim
            transformer_transform = nn.Sequential(
                nn.Linear(self.internal_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, self.internal_dim),
                nn.LayerNorm(self.internal_dim)
            )
            self.transformer_transforms.append(transformer_transform)
        
        # 2. 交叉注意力机制 - 使用internal_dim
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.internal_dim,
            num_heads=6,
            dropout=dropout_rate
        )
        
        # 3. 自适应门控机制 - 使用internal_dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.internal_dim * 2, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[0], self.internal_dim),
            nn.Sigmoid()
        )
        
        self.gate_transformer = nn.Sequential(
            nn.Linear(self.internal_dim * 2, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[0], self.internal_dim),
            nn.Sigmoid()
        )
        
        # 4. 特征融合后的转换 - 使用internal_dim
        self.fusion_transform = nn.Sequential(
            nn.Linear(self.internal_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[0], self.internal_dim)
        )
    
    def forward(self, mlp_features, transformer_features):
        batch_size = mlp_features.shape[0]
        
        # 首先将输入投影到内部维度
        mlp_features = self.input_projection_mlp(mlp_features)
        transformer_features = self.input_projection_transformer(transformer_features)
        
        # 1. 多层特征变换
        mlp_residual = mlp_features
        transformer_residual = transformer_features
        
        for mlp_transform, transformer_transform in zip(self.mlp_transforms, self.transformer_transforms):
            # 残差连接
            mlp_features = mlp_transform(mlp_features) + mlp_residual
            transformer_features = transformer_transform(transformer_features) + transformer_residual
            
            mlp_residual = mlp_features
            transformer_residual = transformer_features
        
        # 2. 交叉注意力
        # 重塑为序列格式: [seq_len, batch, feature_dim]
        mlp_seq = mlp_features.unsqueeze(0)  # [1, batch, feature_dim]
        transformer_seq = transformer_features.unsqueeze(0)  # [1, batch, feature_dim]
        
        # 交叉注意力: transformer特征关注mlp特征
        attn_output, _ = self.cross_attention(
            query=transformer_seq,
            key=mlp_seq,
            value=mlp_seq
        )
        
        # 重塑回原始形状
        attn_output = attn_output.squeeze(0)  # [batch, feature_dim]
        
        # 加入残差连接
        enhanced_transformer = attn_output + transformer_features
        
        # 3. 自适应门控
        # 拼接特征
        concat_features = torch.cat([mlp_features, enhanced_transformer], dim=1)
        
        # 计算门控值
        mlp_gate = self.gate_mlp(concat_features)
        transformer_gate = self.gate_transformer(concat_features)
        
        # 应用门控
        gated_mlp = mlp_features * mlp_gate
        gated_transformer = enhanced_transformer * transformer_gate
        
        # 4. 融合特征
        fused_features = gated_mlp + gated_transformer
        
        # 5. 最终转换
        output = self.fusion_transform(fused_features) + fused_features  # 残差连接
        
        # 6. 投影回原始维度
        output = self.output_projection(output)
        
        return output, {
            'mlp_gate': mlp_gate.mean().item(),
            'transformer_gate': transformer_gate.mean().item()
        }

class AdvancedSpec2MolModel(nn.Module):
    def __init__(self, spectrum_dim,  prop_dim, hdim=2048, n_layers=6, dropout_rate=0.2):
        super(AdvancedSpec2MolModel, self).__init__()
        

        self.kwargs = dict(
            spectrum_dim=spectrum_dim,
            prop_dim=prop_dim,
            hdim=hdim,
            n_layers=n_layers,
            dropout_rate=dropout_rate
        )
        
        self.meta = nn.Parameter(torch.empty(0))
        
        # 1. 质谱特征提取模块
        self.spectrum_encoder = nn.Sequential(
            nn.Linear(spectrum_dim, hdim),
            nn.LayerNorm(hdim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hdim, hdim // 2),
            nn.LayerNorm(hdim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 2. 改进的MLP路径 - 使用ResNet结构
        mlp_blocks = []
        for _ in range(n_layers):
            mlp_blocks.append(
                ResBlock(
                    nn.Sequential(
                        nn.Dropout(p=dropout_rate),
                        nn.Linear(hdim // 2, hdim),
                        nn.LayerNorm(hdim),
                        nn.SiLU(),
                        nn.Linear(hdim, hdim // 2),
                        nn.LayerNorm(hdim // 2)
                    )
                )
            )
        
        self.mlp_path = nn.Sequential(
            *mlp_blocks,
            nn.Linear(hdim // 2, prop_dim),
            nn.SiLU()
        )
        
        # 3. 改进的Transformer路径 - 添加卷积特征提取
        # 先用1D卷积提取局部特征
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),  # 更大的卷积核
            nn.SiLU(),
            nn.MaxPool1d(4, stride=4),  # 更大的池化窗口
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.MaxPool1d(4, stride=4),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.SiLU()
        )
        
        # 计算卷积后的特征维度
        conv_output_dim = spectrum_dim // 16  # 根据具体参数调整
        
        # 投影到transformer维度
        self.conv_projection = nn.Linear(128, hdim // 2)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hdim // 2,
            nhead=8,
            dim_feedforward=hdim,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=4
        )
        
        # 输出投影
        self.transformer_output = nn.Sequential(
            nn.Linear(hdim // 2 * conv_output_dim, hdim),
            nn.LayerNorm(hdim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hdim, prop_dim)
        )
        
        # 4. 改进的特征融合模块
        self.feature_fusion = EnhancedFeatureFusion(
            feature_dim=prop_dim,
            hidden_dims=[hdim, hdim // 2, hdim // 4],
            dropout_rate=dropout_rate
        )
        
        # 5. 输出层
        self.output_layers = nn.Sequential(
            nn.Linear(prop_dim, hdim),
            nn.LayerNorm(hdim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hdim, prop_dim),
            nn.Sigmoid()
        )
        
        # 6. 注意力机制来加强重要特征
        self.attention = nn.Sequential(
            nn.Linear(prop_dim, hdim // 4),
            nn.SiLU(),
            nn.Linear(hdim // 4, prop_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        device = self.meta.device
        x = x.type(torch.FloatTensor).to(device)
        
        # 1. 共享编码器处理输入
        shared_features = self.spectrum_encoder(x)
        
        # 2. MLP路径
        mlp_features = self.mlp_path(shared_features)
        
        # 3. Transformer路径处理
        # 重塑为卷积输入格式 [batch, channels, length]
        batch_size = x.size(0)
        x_conv = x.unsqueeze(1)  # [batch, 1, spectrum_dim]
        
        # 应用卷积层提取特征
        x_conv = self.conv_layers(x_conv)  # [batch, 128, reduced_dim]
        
        # 重塑为Transformer输入格式 [batch, seq_length, features]
        x_conv = x_conv.transpose(1, 2)  # [batch, reduced_dim, 128]
        
        # 投影到Transformer维度
        x_trans = self.conv_projection(x_conv)  # [batch, reduced_dim, hdim//2]
        
        # 应用Transformer编码器
        x_trans = self.transformer_encoder(x_trans)  # [batch, reduced_dim, hdim//2]
        
        # 平铺特征以进行最终输出投影
        x_trans = x_trans.reshape(batch_size, -1)  # [batch, reduced_dim * hdim//2]
        
        # 最终输出投影
        transformer_features = self.transformer_output(x_trans)  # [batch, prop_dim]
        
        # 4. 特征融合
        fused_features, fusion_stats = self.feature_fusion(mlp_features, transformer_features)
        
        # 5. 注意力加权
        attention_weights = self.attention(fused_features)
        weighted_features = fused_features * attention_weights
        
        # 6. 最终输出
        output = self.output_layers(weighted_features)
        
        return output






# new

# class Spec2MolFeaturefusion(nn.Module):
#     def __init__(self, molecule_dim: int, prop_dim: int, hdim: int, n_layers: int):
#         super(Spec2MolFeaturefusion, self).__init__()
#         self.kwargs = dict(
#             molecule_dim=molecule_dim,            
#             prop_dim=prop_dim,
#             hdim=hdim,
#             n_layers=n_layers
#         )
#         self.meta = nn.Parameter(torch.empty(0))
#         self.molecule_dim = molecule_dim
#         self.prop_dim = prop_dim
#         # self.tr = MyEncoder(input_dim=molecule_dim, hidden_dim=1024, num_layers=3, num_heads=8, output_dim=prop_dim)
#         self.tr = MyEncoder(input_dim=molecule_dim, hidden_dim=2048, num_layers=4, num_heads=8, output_dim=prop_dim)
#         self.mlp1 = Mol2SpecSimple(molecule_dim=molecule_dim, prop_dim=prop_dim, hdim=hdim, n_layers=n_layers)
#         self.mlp2 = Mol2SpecSimple(molecule_dim=prop_dim * 2,prop_dim=prop_dim, hdim=hdim, n_layers=n_layers)
#         # self.dropout = nn.Dropout(p=0.1)



#     def forward(self, mz_res: torch.Tensor) -> torch.Tensor:
#         device = self.meta.device
#         # self.tr.eval()
#         # self.mlp1.eval()
#         # self.mlp2.eval()
#         mz_res = mz_res.type(torch.FloatTensor).to(device)
#         mz1 = self.mlp1(mz_res)
#         mz2 = self.tr(mz_res)
#         fused_features = torch.cat((mz1, mz2), dim=1)
#         fused_features = torch.nn.BatchNorm1d(fused_features.size(1)).to(device)(fused_features)
#         # fused_features = self.dropout(fused_features)
#         mol_vec = self.mlp2(fused_features)
#         return mol_vec
    

class Spec2MolFeaturefusion_LSTM(nn.Module):
    def __init__(self, molecule_dim: int, prop_dim: int, hdim: int, n_layers: int):
        super(Spec2MolFeaturefusion_LSTM, self).__init__()
        self.kwargs = dict(
            molecule_dim=molecule_dim,            
            prop_dim=prop_dim,
            hdim=hdim,
            n_layers=n_layers
        )
        self.meta = nn.Parameter(torch.empty(0))
        self.molecule_dim = molecule_dim
        self.prop_dim = prop_dim

        # 使用MLP提取谱图特征
        self.mlp = Mol2SpecSimple(molecule_dim=molecule_dim, prop_dim=prop_dim, hdim=hdim, n_layers=n_layers)

        # 使用BiLSTM提取谱图特征
        self.bilstm = nn.LSTM(input_size=prop_dim * 2, hidden_size=519, num_layers=6, batch_first=True, bidirectional=True)

        # 使用Transformer提取谱图特征
        self.tr = MyEncoder(input_dim=molecule_dim, hidden_dim=1024, num_layers=6, num_heads=8, output_dim=prop_dim)

        # 使用MLP进行最终输出
        self.final_mlp = Mol2SpecSimple(molecule_dim=prop_dim, prop_dim=prop_dim, hdim=hdim, n_layers=n_layers)

        # 使用Batch Normalization和Dropout
        self.bn = nn.BatchNorm1d(prop_dim * 2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, mz_res: torch.Tensor) -> torch.Tensor:
        device = self.meta.device
        mz_res = mz_res.type(torch.FloatTensor).to(device)

        # MLP提取特征
        mz1 = self.mlp(mz_res)
        # print(mz1.shape)
        # Transformer提取特征
        mz2 = self.tr(mz_res)
        # BiLSTM提取特征


        # 特征融合
        fused_features = torch.cat((mz1, mz2), dim=1)
        # print(fused_features.shape)
        fused_features = self.bn(fused_features)
        # fused_features = self.dropout(fused_features)
        fused_features = fused_features.unsqueeze(1)  
        fused_features, _ = self.bilstm(fused_features)
        fused_features = fused_features[:, -1, :]  # 取最后一个时间步的输出
        # print(mz2.shape)


        # 最终输出
        mol_vec = self.final_mlp(fused_features)
        return mol_vec


def test_Spec2MolFeaturefusion():
    # 假设 molecule_dim是特征向量的维度
    spectrum_dim = 6552  # 可根据实际情况调整
    prop_dim = 2062  # 输出维度，例如谱图的m/z预测维度
    hdim = 2048  # 隐藏层维度
    n_layers = 6  # 模型层数
    batch_size = 512  # 批大小

    # 实例化模型
    model = AdvancedSpec2MolModel(spectrum_dim, prop_dim, hdim, n_layers)

    # 生成模拟输入张量
    mz_re = torch.randn(batch_size, spectrum_dim)  # 输入的第一个特征向量，形状为 (batch_size, molecule_dim1)

    # 前向传递
    output = model(mz_re)

    # 输出结果
    print(f"输入 mol_vec1 形状: {mz_re.shape}")
    print(f"模型输出形状: {output.shape}")

    # 检查输出维度是否正确
    assert output.shape == (batch_size, prop_dim), f"输出形状不正确，应为 {(batch_size, prop_dim)}，但得到 {output.shape}"

    print("模型验证通过，输入输出维度正确")
# mlp trans
#     |    |
#
#     mlp
#

# class fusemodel(nn.Module):
#     def __int__(self):
#
#         self.mlp1=Mol2SpecSimple()
#         self.tr=TransformerModel()
#         self.mlp=Mol2SpecSimple()
#
#
#     def forward(self,mol_vec: torch.Tensor) -> torch.Tensor:
#         mol1=self.mlp1(mol_vec)
#         mol2=self.tr(mol_vec)
#         mol=torch.cat([mol1,mol2],1)
#         fuze_mol=torch.nn.BatchNorm1d(num_features=100)
#         output=self.mlp(fuze_mol)
#         return output

# class my_Model(nn.Module):
#     def __init__(self, molecule_dim: int, prop_dim: int, hdim: int, n_layers: int):
#         super().__init__()
#         self.kwargs = dict(
#             molecule_dim=molecule_dim,
#             prop_dim=prop_dim,
#             hdim=hdim,
#             n_layers=n_layers
#         )
#         self.embding=LSTMPredictor(input_size=1,hidden_size=64,num_layers=3,output_size=167).to('cuda:0')
#         self.meta = nn.Parameter(torch.empty(0))
#         self.molecule_dim = molecule_dim
#         self.prop_dim = prop_dim
#         dropout_p = 0.1
#         res_blocks = [
#             ResBlock(
#                 nn.Sequential(
#                     nn.Dropout(p=dropout_p),
#                     nn.Linear(molecule_dim, hdim),
#                     nn.SiLU(),
#                     nn.Linear(hdim, molecule_dim),
#                 )
#             )
#             for _ in range(n_layers)
#         ]
#         self.mlp_layers = nn.Sequential(
#             *res_blocks,
#             nn.Linear(molecule_dim, prop_dim),
#         )
#     def forward(self, x1,x2,x3,x4,x5) -> torch.Tensor:

#         a=self.embding(x2,x3)

#         x1=x1.to(setup_device())
#         x4=x4.to(setup_device())
#         x5=x5.to(setup_device())
#         mol_vec=torch.cat((x1,a, x4,x5), dim=1)
#         mol_vec = mol_vec.type(torch.FloatTensor).to(self.meta.device)
#         mz_res = self.mlp_layers(mol_vec)
#         return mz_res


class Mol2SpecGraph(nn.Module):
    def __init__(self, node_feature_dim: int, graph_feature_dim: int, prop_dim: int, hdim: int, n_layers: int):
        super().__init__()
        self.kwargs = dict(
            node_feature_dim=node_feature_dim,
            graph_feature_dim=graph_feature_dim,
            prop_dim=prop_dim,
            hdim=hdim,
            n_layers=n_layers
        )
        self.meta = nn.Parameter(torch.empty(0))

        gcn_middle_layers = []
        for _ in range(n_layers):
            gcn_middle_layers += [
                (gnn.GCNConv(hdim, hdim), 'x, edge_index -> x'),
                nn.ReLU(inplace=True),
            ]

        self.gcn_layers = gnn.Sequential(
            'x, edge_index, batch',
            [
                (gnn.GCNConv(node_feature_dim, hdim), 'x, edge_index -> x'),
                nn.ReLU(inplace=True),
            ] + gcn_middle_layers + [
                (gnn.global_max_pool, 'x, batch -> x'),
            ]
        )

        dropout_p = 0.1
        res_blocks = [
            ResBlock(
                nn.Sequential(
                    nn.Dropout(p=dropout_p),
                    nn.Linear(hdim + graph_feature_dim, hdim),
                    nn.SiLU(),
                    nn.Linear(hdim, hdim + graph_feature_dim),
                )
            )
            for _ in range(1)
        ]

        self.head_layers = nn.Sequential(
            *res_blocks,
            nn.Linear(hdim + graph_feature_dim, prop_dim),
        )

    def forward(self, gdata: torch_geometric.data.Data) -> torch.Tensor:
        gdata = gdata.to(self.meta.device)
        x = self.gcn_layers(gdata.x, gdata.edge_index, gdata.batch)

        batch_size = x.shape[0]
        frag_levels = gdata.frag_levels.reshape([batch_size, gdata.frag_levels.shape[0] // batch_size])
        adduct_feats = gdata.adduct_feats.reshape([batch_size, gdata.adduct_feats.shape[0] // batch_size])

        x = torch.cat((x, frag_levels, adduct_feats), axis=1)
        x = self.head_layers(x)
        return x


class Mol2SpecEGNN(nn.Module):
    def __init__(self, node_feature_dim: int, graph_feature_dim: int, prop_dim: int, hdim: int, edge_dim: int, n_layers: int):
        super().__init__()
        self.kwargs = dict(
            node_feature_dim=node_feature_dim,
            graph_feature_dim=graph_feature_dim,
            prop_dim=prop_dim,
            hdim=hdim,
            edge_dim=edge_dim,
            n_layers=n_layers,
        )
        self.meta = nn.Parameter(torch.empty(0))

        self.egnn = egnn.EGNN(
            in_node_nf=node_feature_dim,
            hidden_nf=hdim,
            out_node_nf=1024,
            in_edge_nf=edge_dim,
            n_layers=n_layers
            )

        self.pool_layers = gnn.Sequential('x, batch', [
            (gnn.global_max_pool, 'x, batch -> x'),
            ])

        # Would have liked training more than 1 layer, but my current setup is sooo slow
        dropout_p = 0.1
        res_blocks = [
            ResBlock(
                nn.Sequential(
                    nn.Dropout(p=dropout_p),
                    nn.Linear(hdim + graph_feature_dim, hdim),
                    nn.SiLU(),
                    nn.Linear(hdim, hdim + graph_feature_dim),
                )
            )
            for _ in range(1)
        ]

        self.head_layers = nn.Sequential(
            *res_blocks,
            nn.Linear(hdim + graph_feature_dim, prop_dim),
        )

    def forward(self, gdata: torch_geometric.data.Data,x1) -> torch.Tensor:
        gdata = gdata.to(self.meta.device)
        x1=x1.to(self.meta.device)
        x, _ = self.egnn(gdata.x, gdata.pos, gdata.edge_index, gdata.edge_attr)
        x = self.pool_layers(x, gdata.batch)

        batch_size = x.shape[0]
        frag_levels = gdata.frag_levels.reshape([batch_size, gdata.frag_levels.shape[0] // batch_size])
        adduct_feats = gdata.adduct_feats.reshape([batch_size, gdata.adduct_feats.shape[0] // batch_size])
        x = torch.cat((x, x1,frag_levels, adduct_feats), axis=1)
        x=x.to(torch.float)
        x = self.head_layers(x)
        return x


if __name__ == '__main__':

    test_Spec2MolFeaturefusion()
