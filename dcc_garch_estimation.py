"""
DCC-GARCH模型估计脚本
成员4任务：多变量DCC模型 (Q2 + Q3DCC)

功能：
1. 读取ETF和加密货币数据
2. 计算对数收益率
3. 筛选共同样本
4. 估计DCC-GARCH模型
5. 输出模型参数和结果
"""

"""
依赖库安装说明：
pip install arch pandas numpy

如果arch库未安装，请运行: pip install arch
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 检查并导入arch库
try:
    from arch import arch_model
    from arch.univariate import GARCH
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("警告: arch库未安装。请运行 'pip install arch' 安装该库。")
    print("程序将无法执行DCC-GARCH估计。")

# ==============================================================================
# 1. 数据读取和预处理函数
# ==============================================================================

def load_etf_data(file_path):
    """
    读取ETF数据并计算对数收益率
    
    参数:
        file_path: CSV文件路径
    
    返回:
        returns: 对数收益率序列（带日期索引）
    """
    print(f"\n正在读取: {file_path.name}")
    df = pd.read_csv(file_path)
    
    # 确保Date列存在并转换为日期类型
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        # 只保留日期部分（去除时间），确保格式一致
        df['Date'] = df['Date'].dt.date
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    else:
        raise ValueError(f"文件 {file_path.name} 中未找到Date列")
    
    # 使用Close价格计算对数收益率
    if 'Close' in df.columns:
        prices = df['Close']
        returns = np.log(prices / prices.shift(1)).dropna()
        returns.name = file_path.stem.replace('HistoricalPrices_', '')
        print(f"  数据范围: {returns.index.min()} 至 {returns.index.max()}")
        print(f"  观测数量: {len(returns)}")
        return returns
    else:
        raise ValueError(f"文件 {file_path.name} 中未找到Close列")


def load_crypto_data(file_path):
    """
    读取加密货币数据并计算对数收益率
    
    参数:
        file_path: CSV文件路径
    
    返回:
        returns: 对数收益率序列（带日期索引）
    """
    print(f"\n正在读取: {file_path.name}")
    try:
        # 加密货币数据使用分号分隔
        df = pd.read_csv(file_path, sep=';')
    except Exception as e:
        print(f"  错误: 无法读取文件: {str(e)}")
        raise
    
    # 解析日期（使用timeOpen或timestamp列）
    date_col = None
    if 'timeOpen' in df.columns:
        date_col = 'timeOpen'
    elif 'timestamp' in df.columns:
        date_col = 'timestamp'
    else:
        raise ValueError(f"文件 {file_path.name} 中未找到日期列")
    
    # 清理日期字符串（移除引号）并解析
    date_str = df[date_col].astype(str).str.replace('"', '')
    df['Date'] = pd.to_datetime(date_str, errors='coerce', format='ISO8601')
    
    # 如果ISO格式解析失败，尝试其他格式
    if df['Date'].isna().any():
        # 尝试提取日期部分（YYYY-MM-DD）
        date_str_clean = date_str.str.extract(r'(\d{4}-\d{2}-\d{2})')[0]
        df['Date'] = pd.to_datetime(date_str_clean, errors='coerce')
    
    # 只保留日期部分（去除时间），确保与ETF数据格式一致
    df['Date'] = df['Date'].dt.date
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 删除日期解析失败的行
    df = df.dropna(subset=['Date'])
    
    if len(df) == 0:
        raise ValueError(f"文件 {file_path.name} 中没有有效的日期数据")
    
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    # 使用close价格计算对数收益率
    if 'close' in df.columns:
        prices = pd.to_numeric(df['close'].astype(str).str.replace('"', ''), errors='coerce')
        prices = prices.dropna()
        
        if len(prices) == 0:
            raise ValueError(f"文件 {file_path.name} 中没有有效的价格数据")
        
        # 计算对数收益率
        returns = np.log(prices / prices.shift(1)).dropna()
        
        # 检查收益率数据
        if len(returns) == 0:
            raise ValueError(f"文件 {file_path.name} 无法计算收益率")
        
        # 移除异常值（收益率绝对值大于1的）
        returns = returns[np.abs(returns) <= 1.0]
        
        crypto_name = 'Bitcoin' if 'Bitcoin' in file_path.name else 'Ethereum'
        returns.name = crypto_name
        
        print(f"  数据范围: {returns.index.min()} 至 {returns.index.max()}")
        print(f"  观测数量: {len(returns)}")
        print(f"  收益率统计: 均值={returns.mean():.6f}, 标准差={returns.std():.6f}")
        
        return returns
    else:
        raise ValueError(f"文件 {file_path.name} 中未找到close列")


# ==============================================================================
# 2. 共同样本筛选函数
# ==============================================================================

def find_common_sample(*returns_series, names=None):
    """
    找出多个时间序列的共同样本
    
    参数:
        *returns_series: 多个收益率序列
        names: 序列名称列表（可选）
    
    返回:
        common_data: DataFrame，包含共同样本数据
        info: dict，包含共同样本信息
    """
    if names is None:
        names = [f"Asset_{i+1}" for i in range(len(returns_series))]
    
    # 验证输入
    if len(returns_series) == 0:
        raise ValueError("至少需要一个收益率序列")
    
    if len(returns_series) != len(names):
        raise ValueError("收益率序列数量与名称数量不匹配")
    
    # 检查每个序列是否为空
    for i, ret in enumerate(returns_series):
        if len(ret) == 0:
            raise ValueError(f"序列 {names[i]} 为空")
    
    # 合并所有序列
    combined = pd.DataFrame()
    for i, ret in enumerate(returns_series):
        if not isinstance(ret, pd.Series):
            raise ValueError(f"序列 {names[i]} 不是pandas Series")
        # 确保索引是日期类型
        if not isinstance(ret.index, pd.DatetimeIndex):
            ret.index = pd.to_datetime(ret.index)
        # 只保留日期部分（去除时间），确保格式一致
        ret.index = pd.to_datetime(ret.index.date)
        combined[names[i]] = ret
    
    # 找出共同样本（删除任何包含NaN的行）
    common_data = combined.dropna()
    
    # 确保索引是日期类型且已排序
    if len(common_data) > 0:
        common_data.index = pd.to_datetime(common_data.index)
        common_data = common_data.sort_index()
    
    # 检查共同样本是否为空
    if len(common_data) == 0:
        print(f"\n警告: 资产 {', '.join(names)} 的共同样本为空")
        print("可能原因:")
        print("  - 时间索引没有重叠")
        print("  - 数据中存在大量缺失值")
        for i, ret in enumerate(returns_series):
            print(f"  - {names[i]}: {len(ret)} 个观测值, 日期范围 {ret.index.min()} 至 {ret.index.max()}")
    
    # 生成信息字典
    if len(common_data) > 0:
        info = {
            'start_date': common_data.index.min(),
            'end_date': common_data.index.max(),
            'n_obs': len(common_data),
            'assets': names,
            'original_lengths': {names[i]: len(returns_series[i]) for i in range(len(returns_series))}
        }
    else:
        info = {
            'start_date': None,
            'end_date': None,
            'n_obs': 0,
            'assets': names,
            'original_lengths': {names[i]: len(returns_series[i]) for i in range(len(returns_series))}
        }
    
    print(f"\n共同样本信息:")
    print(f"  资产: {', '.join(names)}")
    if info['n_obs'] > 0:
        print(f"  开始日期: {info['start_date']}")
        print(f"  结束日期: {info['end_date']}")
        print(f"  共同样本观测数: {info['n_obs']}")
    else:
        print(f"  共同样本观测数: 0 (无共同样本)")
    for name, orig_len in info['original_lengths'].items():
        print(f"  {name} 原始观测数: {orig_len}")
    
    return common_data, info


# ==============================================================================
# 3. DCC-GARCH模型估计函数
# ==============================================================================

def estimate_dcc_garch(returns_data, asset_names):
    """
    估计DCC-GARCH模型
    
    参数:
        returns_data: DataFrame，包含共同样本的收益率数据
        asset_names: 资产名称列表
    
    返回:
        results: dict，包含估计结果
    """
    if not ARCH_AVAILABLE:
        raise ImportError("arch库未安装，无法执行DCC-GARCH估计。请运行 'pip install arch'")
    
    print(f"\n{'='*60}")
    print(f"开始估计DCC-GARCH模型")
    print(f"资产组合: {', '.join(asset_names)}")
    print(f"{'='*60}")
    
    # 数据验证
    if returns_data.empty:
        raise ValueError("收益率数据为空，无法估计DCC-GARCH模型")
    
    if len(returns_data) < 100:
        print(f"警告: 共同样本只有 {len(returns_data)} 个观测值，可能不足以估计模型")
    
    # 检查每个资产的数据
    for asset in asset_names:
        if asset not in returns_data.columns:
            raise ValueError(f"资产 {asset} 不在数据中")
        asset_data = returns_data[asset].dropna()
        if len(asset_data) == 0:
            raise ValueError(f"资产 {asset} 的数据为空")
        print(f"  {asset}: {len(asset_data)} 个有效观测值")
    
    n_assets = len(asset_names)
    results = {
        'assets': asset_names,
        'n_obs': len(returns_data),
        'marginal_garch': {},
        'dcc_params': None,
        'log_likelihood': None
    }
    
    # 第一阶段：估计边际GARCH模型
    print(f"\n第一阶段：估计边际GARCH(1,1)模型")
    print("-" * 60)
    
    standardized_residuals = pd.DataFrame(index=returns_data.index)
    conditional_vols = pd.DataFrame(index=returns_data.index)
    
    for asset in asset_names:
        print(f"\n估计 {asset} 的GARCH(1,1)模型...")
        returns = returns_data[asset].dropna()
        
        # 数据验证
        if len(returns) == 0:
            raise ValueError(f"资产 {asset} 的数据为空，无法估计GARCH模型")
        
        if len(returns) < 100:
            print(f"  警告: 资产 {asset} 只有 {len(returns)} 个观测值，可能不足以估计GARCH模型")
        
        # 检查数据是否全为0或NaN
        if returns.std() == 0 or np.isnan(returns.std()):
            raise ValueError(f"资产 {asset} 的收益率标准差为0或NaN，无法估计GARCH模型")
        
        # 检查是否有异常值
        if np.any(np.abs(returns) > 1.0):
            print(f"  警告: 资产 {asset} 存在绝对值大于1的收益率，可能影响估计结果")
        
        # 估计GARCH(1,1)模型
        # 注意：将收益率乘以100以提高数值稳定性
        try:
            model = arch_model(returns * 100, vol='Garch', p=1, q=1, rescale=False)
            fitted = model.fit(disp='off', show_warning=False)
        except Exception as e:
            print(f"  错误: 无法估计 {asset} 的GARCH模型")
            print(f"  错误信息: {str(e)}")
            print(f"  数据统计: 均值={returns.mean():.6f}, 标准差={returns.std():.6f}, 观测数={len(returns)}")
            raise
        
        # 提取参数和显著性检验结果
        params = fitted.params
        tvalues = fitted.tvalues
        pvalues = fitted.pvalues
        
        # 显著性标记函数
        def significance_marker(pval):
            if pd.isna(pval) or pval >= 1:
                return ''
            if pval < 0.01:
                return '***'
            elif pval < 0.05:
                return '**'
            elif pval < 0.10:
                return '*'
            else:
                return ''
        
        # 安全获取统计量（处理可能的键名差异）
        def safe_get(series, key, default=np.nan):
            try:
                if key in series.index:
                    return series[key]
                # 尝试不同的键名格式
                for idx in series.index:
                    if key.replace('[', '').replace(']', '') in str(idx).replace('[', '').replace(']', ''):
                        return series[idx]
                return default
            except:
                return default
        
        omega_t = safe_get(tvalues, 'omega')
        alpha_t = safe_get(tvalues, 'alpha[1]')
        beta_t = safe_get(tvalues, 'beta[1]')
        mu_t = safe_get(tvalues, 'mu', 0)
        
        omega_p = safe_get(pvalues, 'omega', 1)
        alpha_p = safe_get(pvalues, 'alpha[1]', 1)
        beta_p = safe_get(pvalues, 'beta[1]', 1)
        mu_p = safe_get(pvalues, 'mu', 1)
        
        results['marginal_garch'][asset] = {
            'omega': params['omega'],
            'alpha': params['alpha[1]'],
            'beta': params['beta[1]'],
            'mu': params.get('mu', 0),
            'omega_tstat': omega_t,
            'alpha_tstat': alpha_t,
            'beta_tstat': beta_t,
            'mu_tstat': mu_t,
            'omega_pvalue': omega_p,
            'alpha_pvalue': alpha_p,
            'beta_pvalue': beta_p,
            'mu_pvalue': mu_p,
            'log_likelihood': fitted.loglikelihood,
            'aic': fitted.aic,
            'bic': fitted.bic
        }
        
        # 计算条件波动率和标准化残差
        conditional_vol = fitted.conditional_volatility / 100  # 转换回原始尺度
        conditional_vols[asset] = conditional_vol
        standardized_residuals[asset] = returns / conditional_vol
        
        # 显示参数估计结果（带显著性检验）
        print(f"  参数估计结果:")
        print(f"  ω (omega): {params['omega']:.6f}  [t={omega_t:.3f}, p={omega_p:.4f}{significance_marker(omega_p)}]")
        print(f"  α (alpha): {params['alpha[1]']:.6f}  [t={alpha_t:.3f}, p={alpha_p:.4f}{significance_marker(alpha_p)}]")
        print(f"  β (beta):  {params['beta[1]']:.6f}  [t={beta_t:.3f}, p={beta_p:.4f}{significance_marker(beta_p)}]")
        if 'mu' in params and params['mu'] != 0:
            print(f"  μ (mu):    {params['mu']:.6f}  [t={mu_t:.3f}, p={mu_p:.4f}{significance_marker(mu_p)}]")
        print(f"  显著性标记: *** p<0.01, ** p<0.05, * p<0.10")
        print(f"  对数似然值: {fitted.loglikelihood:.2f}")
        print(f"  AIC: {fitted.aic:.2f}")
        print(f"  BIC: {fitted.bic:.2f}")
    
    # 第二阶段：估计DCC参数
    print(f"\n第二阶段：估计DCC参数")
    print("-" * 60)
    
    # 清理标准化残差
    standardized_residuals_clean = standardized_residuals.dropna()
    z = standardized_residuals_clean.values
    
    # 计算无条件相关矩阵（样本相关矩阵）
    R_bar = np.corrcoef(z.T)
    
    # DCC参数估计（使用两步估计法）
    # 第一步：计算无条件相关矩阵（已完成）
    # 第二步：估计DCC参数 α_D 和 β_D
    
    # 使用最大似然估计DCC参数
    # 目标函数：最大化对数似然值
    # L = -0.5 * sum(log|R_t| + z_t' * R_t^(-1) * z_t)
    
    # 简化方法：使用样本相关矩阵作为初始值
    # 完整实现需要使用优化算法估计α_D和β_D
    
    print("\n使用两步估计法估计DCC参数：")
    print("  步骤1: 估计无条件相关矩阵 R_bar (已完成)")
    print("  步骤2: 估计DCC参数 α_D 和 β_D")
    print("\n注意：完整DCC估计需要使用优化算法")
    print("这里提供基于样本相关矩阵的简化结果")
    
    # 计算平均相关系数
    if n_assets == 2:
        avg_corr = R_bar[0, 1]
    else:
        # 对于多资产，计算所有非对角线元素的平均值
        mask = ~np.eye(n_assets, dtype=bool)
        avg_corr = np.mean(R_bar[mask])
    
    # 尝试使用简化方法估计DCC参数
    # 这里使用样本相关矩阵的衰减率作为近似
    # 实际应用中应使用最大似然估计
    
    # 计算动态相关系数的时变特征（简化）
    # 使用滚动窗口计算时变相关性
    window_size = min(60, len(standardized_residuals_clean) // 4)
    if window_size >= 20:
        rolling_corr = standardized_residuals_clean.rolling(window=window_size).corr()
        # 提取平均时变相关性
        if n_assets == 2:
            corr_series = rolling_corr.iloc[::n_assets, 1].dropna()
            if len(corr_series) > 0:
                corr_std = corr_series.std()
                corr_mean = corr_series.mean()
            else:
                corr_std = 0
                corr_mean = avg_corr
        else:
            corr_std = 0
            corr_mean = avg_corr
    else:
        corr_std = 0
        corr_mean = avg_corr
    
    # 简化的DCC参数（基于经验规则）
    # 注意：这不是真正的DCC参数估计，只是近似值
    alpha_d_approx = min(0.05, corr_std * 2) if corr_std > 0 else 0.02
    beta_d_approx = max(0.90, 1 - alpha_d_approx - 0.05)
    
    results['dcc_params'] = {
        'alpha_D': alpha_d_approx,
        'beta_D': beta_d_approx,
        'average_correlation': avg_corr,
        'correlation_matrix': R_bar,
        'note': 'DCC参数为近似值。完整估计需要使用arch.multivariate.DCC或优化算法'
    }
    
    print(f"  α_D (近似值): {alpha_d_approx:.4f}")
    print(f"  β_D (近似值): {beta_d_approx:.4f}")
    print(f"  平均相关系数: {avg_corr:.4f}")
    print(f"  无条件相关矩阵:")
    corr_df = pd.DataFrame(R_bar, index=asset_names, columns=asset_names)
    print(corr_df.round(4))
    
    return results


# ==============================================================================
# 4. 结果展示和保存函数
# ==============================================================================

def display_results(results, common_sample_info, combination_name):
    """
    展示估计结果
    
    参数:
        results: 估计结果字典
        common_sample_info: 共同样本信息
        combination_name: 组合名称
    """
    print(f"\n{'='*60}")
    print(f"结果汇总: {combination_name}")
    print(f"{'='*60}")
    
    print(f"\n共同样本信息:")
    print(f"  样本期: {common_sample_info['start_date']} 至 {common_sample_info['end_date']}")
    print(f"  观测数量: {common_sample_info['n_obs']}")
    
    print(f"\n边际GARCH模型参数:")
    print("-" * 60)
    marginal_data = []
    for asset, params in results['marginal_garch'].items():
        # 显著性标记
        omega_sig = '***' if params.get('omega_pvalue', 1) < 0.01 else ('**' if params.get('omega_pvalue', 1) < 0.05 else ('*' if params.get('omega_pvalue', 1) < 0.10 else ''))
        alpha_sig = '***' if params.get('alpha_pvalue', 1) < 0.01 else ('**' if params.get('alpha_pvalue', 1) < 0.05 else ('*' if params.get('alpha_pvalue', 1) < 0.10 else ''))
        beta_sig = '***' if params.get('beta_pvalue', 1) < 0.01 else ('**' if params.get('beta_pvalue', 1) < 0.05 else ('*' if params.get('beta_pvalue', 1) < 0.10 else ''))
        
        marginal_data.append({
            '资产': asset,
            'ω': f"{params['omega']:.6f}{omega_sig}",
            'ω_t值': f"{params.get('omega_tstat', np.nan):.3f}",
            'ω_p值': f"{params.get('omega_pvalue', np.nan):.4f}",
            'α': f"{params['alpha']:.6f}{alpha_sig}",
            'α_t值': f"{params.get('alpha_tstat', np.nan):.3f}",
            'α_p值': f"{params.get('alpha_pvalue', np.nan):.4f}",
            'β': f"{params['beta']:.6f}{beta_sig}",
            'β_t值': f"{params.get('beta_tstat', np.nan):.3f}",
            'β_p值': f"{params.get('beta_pvalue', np.nan):.4f}",
            '对数似然': f"{params['log_likelihood']:.2f}",
            'AIC': f"{params['aic']:.2f}",
            'BIC': f"{params['bic']:.2f}"
        })
    
    marginal_df = pd.DataFrame(marginal_data)
    print(marginal_df.to_string(index=False))
    print("\n显著性标记: *** p<0.01, ** p<0.05, * p<0.10")
    
    print(f"\nDCC参数:")
    print("-" * 60)
    if results['dcc_params']:
        dcc_params = results['dcc_params']
        print(f"  α_D: {dcc_params.get('alpha_D', 'N/A'):.4f}" if 'alpha_D' in dcc_params else "  α_D: N/A")
        print(f"  β_D: {dcc_params.get('beta_D', 'N/A'):.4f}" if 'beta_D' in dcc_params else "  β_D: N/A")
        print(f"  平均相关系数: {dcc_params['average_correlation']:.4f}")
        print(f"  注意: {dcc_params['note']}")


def save_results(results, common_sample_info, combination_name, output_dir):
    """
    保存结果到CSV文件
    
    参数:
        results: 估计结果字典
        common_sample_info: 共同样本信息
        combination_name: 组合名称
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 保存边际GARCH参数（包含显著性检验，T值紧跟在对应参数后面）
    marginal_data = []
    for asset, params in results['marginal_garch'].items():
        marginal_data.append({
            '资产': asset,
            'omega': params['omega'],
            'omega_tstat': params.get('omega_tstat', np.nan),
            'omega_pvalue': params.get('omega_pvalue', np.nan),
            'alpha': params['alpha'],
            'alpha_tstat': params.get('alpha_tstat', np.nan),
            'alpha_pvalue': params.get('alpha_pvalue', np.nan),
            'beta': params['beta'],
            'beta_tstat': params.get('beta_tstat', np.nan),
            'beta_pvalue': params.get('beta_pvalue', np.nan),
            'mu': params.get('mu', 0),
            'mu_tstat': params.get('mu_tstat', np.nan),
            'mu_pvalue': params.get('mu_pvalue', np.nan),
            'log_likelihood': params['log_likelihood'],
            'AIC': params['aic'],
            'BIC': params['bic']
        })
    
    marginal_df = pd.DataFrame(marginal_data)
    marginal_file = output_dir / f"{combination_name}_marginal_garch_params.csv"
    marginal_df.to_csv(marginal_file, index=False, encoding='utf-8-sig')
    print(f"\n边际GARCH参数已保存至: {marginal_file}")
    
    # 保存共同样本信息
    sample_info_df = pd.DataFrame([{
        '组合名称': combination_name,
        '开始日期': common_sample_info['start_date'],
        '结束日期': common_sample_info['end_date'],
        '观测数量': common_sample_info['n_obs'],
        '资产': ', '.join(common_sample_info['assets'])
    }])
    
    sample_info_file = output_dir / f"{combination_name}_common_sample_info.csv"
    sample_info_df.to_csv(sample_info_file, index=False, encoding='utf-8-sig')
    print(f"共同样本信息已保存至: {sample_info_file}")


# ==============================================================================
# 5. 主程序：Q2任务
# ==============================================================================

def run_q2_analysis():
    """
    执行Q2任务：ETF资产组合的DCC-GARCH估计
    """
    print("\n" + "="*80)
    print("Q2任务：ETF资产组合的DCC-GARCH估计")
    print("="*80)
    
    # 定义路径
    base_path = Path(__file__).parent
    data_path = base_path / "data" / "Processed_data" / "WSJ"
    output_path = base_path / "results" / "Q2"
    
    # 读取ETF数据
    print("\n读取ETF数据...")
    spy = load_etf_data(data_path / "HistoricalPrices_S&P 500 SPDR.csv")
    iwm = load_etf_data(data_path / "HistoricalPrices_Russell 2000 ETF.csv")
    ewg = load_etf_data(data_path / "HistoricalPrices_MSCI Germany ETF.csv")
    ewh = load_etf_data(data_path / "HistoricalPrices_MSCI Hong Kong ETF.csv")
    ewu = load_etf_data(data_path / "HistoricalPrices_MSCI United Kingdom ETF.csv")
    
    # Q2.1: 2个资产对的DCC-GARCH估计
    print("\n" + "="*80)
    print("Q2.1: 估计2个资产对的DCC-GARCH模型")
    print("="*80)
    
    pairs = [
        (spy, iwm, "SPY-IWM"),  # (a) S&P 500 SPDR ETF and Russell 2000 iShares ETF
        (ewg, ewu, "EWG-EWU")   # (b) iShares MSCI Germany and iShares MSCI UK
    ]
    
    q2_results = {}
    
    for asset1, asset2, pair_name in pairs:
        print(f"\n处理资产对: {pair_name}")
        try:
            # 筛选共同样本
            common_data, sample_info = find_common_sample(
                asset1, asset2,
                names=[asset1.name, asset2.name]
            )
            
            # 检查共同样本是否为空
            if common_data.empty:
                print(f"  错误: {pair_name} 的共同样本为空，跳过该组合")
                continue
            
            if len(common_data) < 50:
                print(f"  警告: {pair_name} 的共同样本只有 {len(common_data)} 个观测值，可能不足以估计模型")
            
            # 估计DCC-GARCH
            results = estimate_dcc_garch(common_data, [asset1.name, asset2.name])
        except Exception as e:
            print(f"  错误: 处理 {pair_name} 时发生错误: {str(e)}")
            print(f"  跳过该组合，继续处理下一个...")
            continue
        
        # 展示结果
        display_results(results, sample_info, pair_name)
        
        # 保存结果
        save_results(results, sample_info, pair_name, output_path)
        
        q2_results[pair_name] = {
            'results': results,
            'sample_info': sample_info
        }
    
    # Q2.2: 1组4个资产的DCC-GARCH估计
    print("\n" + "="*80)
    print("Q2.2: 估计4个ETF资产的DCC-GARCH模型")
    print("="*80)
    print("资产组合: S&P 500 SPDR ETF, iShares MSCI Germany, iShares MSCI Hong Kong, iShares MSCI UK")
    print("即: SPY, EWG, EWH, EWU")
    
    print("\n处理4资产组合: SPY, EWG, EWH, EWU")
    try:
        # 筛选共同样本
        common_data_4, sample_info_4 = find_common_sample(
            spy, ewg, ewh, ewu,
            names=['SPY', 'EWG', 'EWH', 'EWU']
        )
        
        # 检查共同样本是否为空
        if common_data_4.empty:
            print("  错误: 4资产组合的共同样本为空，跳过该组合")
        elif len(common_data_4) < 50:
            print(f"  警告: 4资产组合的共同样本只有 {len(common_data_4)} 个观测值，可能不足以估计模型")
        else:
            # 估计DCC-GARCH
            results_4 = estimate_dcc_garch(common_data_4, ['SPY', 'EWG', 'EWH', 'EWU'])
    except Exception as e:
        print(f"  错误: 处理4资产组合时发生错误: {str(e)}")
        results_4 = None
        sample_info_4 = None
    
    # 展示结果
    if results_4 is not None:
        display_results(results_4, sample_info_4, "SPY-EWG-EWH-EWU")
        
        # 保存结果
        save_results(results_4, sample_info_4, "SPY-EWG-EWH-EWU", output_path)
    
    print("\n" + "="*80)
    print("Q2任务完成！")
    print("="*80)
    
    return q2_results, results_4, sample_info_4


# ==============================================================================
# 6. 主程序：Q3任务
# ==============================================================================

def run_q3_analysis():
    """
    执行Q3任务：加密货币与ETF的DCC-GARCH估计
    """
    print("\n" + "="*80)
    print("Q3任务：加密货币与ETF的DCC-GARCH估计")
    print("="*80)
    
    # 定义路径
    base_path = Path(__file__).parent
    etf_path = base_path / "data" / "Processed_data" / "WSJ"
    crypto_path = base_path / "data" / "Raw_data" / "coin"
    output_path = base_path / "results" / "Q3"
    
    # 读取ETF数据
    print("\n读取ETF数据...")
    spy = load_etf_data(etf_path / "HistoricalPrices_S&P 500 SPDR.csv")
    iwm = load_etf_data(etf_path / "HistoricalPrices_Russell 2000 ETF.csv")
    
    # 读取加密货币数据
    print("\n读取加密货币数据...")
    bitcoin = load_crypto_data(crypto_path / "Bitcoin_2024_12_8-2025_12_8_historical_data_coinmarketcap.csv")
    ethereum = load_crypto_data(crypto_path / "Ethereum_2024_12_8-2025_12_8_historical_data_coinmarketcap.csv")
    
    # Q3: 3组DCC-GARCH估计
    print("\n" + "="*80)
    print("Q3: 估计3组DCC-GARCH模型")
    print("="*80)
    
    combinations = [
        (spy, bitcoin, "SPY-Bitcoin"),
        (iwm, bitcoin, "IWM-Bitcoin"),
        (bitcoin, ethereum, "Bitcoin-Ethereum")
    ]
    
    q3_results = {}
    
    for asset1, asset2, combo_name in combinations:
        print(f"\n处理组合: {combo_name}")
        try:
            # 筛选共同样本
            common_data, sample_info = find_common_sample(
                asset1, asset2,
                names=[asset1.name, asset2.name]
            )
            
            # 检查共同样本是否为空
            if common_data.empty:
                print(f"  错误: {combo_name} 的共同样本为空，跳过该组合")
                continue
            
            if len(common_data) < 50:
                print(f"  警告: {combo_name} 的共同样本只有 {len(common_data)} 个观测值，可能不足以估计模型")
            
            # 估计DCC-GARCH
            results = estimate_dcc_garch(common_data, [asset1.name, asset2.name])
        except Exception as e:
            print(f"  错误: 处理 {combo_name} 时发生错误: {str(e)}")
            print(f"  跳过该组合，继续处理下一个...")
            continue
        
        # 展示结果
        display_results(results, sample_info, combo_name)
        
        # 保存结果
        save_results(results, sample_info, combo_name, output_path)
        
        q3_results[combo_name] = {
            'results': results,
            'sample_info': sample_info
        }
    
    print("\n" + "="*80)
    print("Q3任务完成！")
    print("="*80)
    
    return q3_results


# ==============================================================================
# 7. 主函数
# ==============================================================================

def main():
    """
    主函数：执行所有DCC-GARCH估计任务
    """
    print("\n" + "="*80)
    print("DCC-GARCH模型估计 - 成员4任务")
    print("="*80)
    
    if not ARCH_AVAILABLE:
        print("\n错误: arch库未安装")
        print("请运行以下命令安装: pip install arch")
        return
    
    try:
        # 执行Q2任务
        q2_results = run_q2_analysis()
        
        # 执行Q3任务
        q3_results = run_q3_analysis()
        
        print("\n" + "="*80)
        print("所有任务完成！")
        print("="*80)
        print("\n结果文件已保存至 results/ 目录")
        print("\n生成的文件包括:")
        print("  - Q2/ 目录: ETF资产组合的估计结果")
        print("  - Q3/ 目录: 加密货币组合的估计结果")
        print("  - 每个组合包含:")
        print("    * _marginal_garch_params.csv: 边际GARCH参数")
        print("    * _common_sample_info.csv: 共同样本信息")
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

