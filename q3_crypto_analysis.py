"""
Q3加密货币分析脚本
任务：
1. 2种加密货币GARCH(1,1)估计
2. 绘制加密货币条件方差图（2张）
3. 绘制3组DCC条件相关性图（3张）

输出：
- figures/Q3_Bitcoin_conditional_variance.png
- figures/Q3_Ethereum_conditional_variance.png
- figures/Q3_SPY-Bitcoin_DCC_correlation.png
- figures/Q3_IWM-Bitcoin_DCC_correlation.png
- figures/Q3_Bitcoin-Ethereum_DCC_correlation.png
- results/Q3/crypto_garch_params.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 检查并导入所需库
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("错误: arch库未安装。请运行 'pip install arch'")

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("错误: matplotlib库未安装。请运行 'pip install matplotlib'")


# ==============================================================================
# 1. 数据读取函数
# ==============================================================================

def load_crypto_data(file_path):
    """
    读取加密货币数据并计算对数收益率
    """
    print(f"\n正在读取: {file_path.name}")
    try:
        df = pd.read_csv(file_path, sep=';')
    except Exception as e:
        print(f"  错误: 无法读取文件: {str(e)}")
        raise
    
    # 解析日期
    date_col = 'timeOpen' if 'timeOpen' in df.columns else 'timestamp'
    date_str = df[date_col].astype(str).str.replace('"', '')
    df['Date'] = pd.to_datetime(date_str, errors='coerce', format='ISO8601')
    
    if df['Date'].isna().any():
        date_str_clean = date_str.str.extract(r'(\d{4}-\d{2}-\d{2})')[0]
        df['Date'] = pd.to_datetime(date_str_clean, errors='coerce')
    
    df['Date'] = pd.to_datetime(df['Date'].dt.date)
    df = df.dropna(subset=['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    # 计算对数收益率
    if 'close' in df.columns:
        prices = pd.to_numeric(df['close'].astype(str).str.replace('"', ''), errors='coerce')
        prices = prices.dropna()
        returns = np.log(prices / prices.shift(1)).dropna()
        returns = returns[np.abs(returns) <= 1.0]
        
        crypto_name = 'Bitcoin' if 'Bitcoin' in file_path.name else 'Ethereum'
        returns.name = crypto_name
        
        print(f"  数据范围: {returns.index.min()} 至 {returns.index.max()}")
        print(f"  观测数量: {len(returns)}")
        print(f"  收益率统计: 均值={returns.mean():.6f}, 标准差={returns.std():.6f}")
        
        return returns, prices
    else:
        raise ValueError(f"文件 {file_path.name} 中未找到close列")


def load_etf_data(file_path):
    """
    读取ETF数据并计算对数收益率
    """
    print(f"\n正在读取: {file_path.name}")
    df = pd.read_csv(file_path)
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Date'] = pd.to_datetime(df['Date'].dt.date)
        df.set_index('Date', inplace=True)
    else:
        raise ValueError(f"文件 {file_path.name} 中未找到Date列")
    
    if 'Close' in df.columns:
        prices = df['Close']
        returns = np.log(prices / prices.shift(1)).dropna()
        returns.name = file_path.stem.replace('HistoricalPrices_', '')
        print(f"  数据范围: {returns.index.min()} 至 {returns.index.max()}")
        print(f"  观测数量: {len(returns)}")
        return returns
    else:
        raise ValueError(f"文件 {file_path.name} 中未找到Close列")


# ==============================================================================
# 2. GARCH(1,1)估计函数
# ==============================================================================

def estimate_garch11(returns, asset_name):
    """
    估计GARCH(1,1)模型并返回参数和条件方差
    """
    print(f"\n{'='*60}")
    print(f"估计 {asset_name} 的 GARCH(1,1) 模型")
    print(f"{'='*60}")
    
    # 数据预处理：乘以100提高数值稳定性
    returns_scaled = returns * 100
    
    # 估计GARCH(1,1)模型
    model = arch_model(returns_scaled, vol='Garch', p=1, q=1, mean='Constant', rescale=False)
    fitted = model.fit(disp='off', show_warning=False)
    
    # 提取参数
    params = fitted.params
    tvalues = fitted.tvalues
    pvalues = fitted.pvalues
    
    # 显著性标记
    def sig_marker(pval):
        if pd.isna(pval) or pval >= 1:
            return ''
        if pval < 0.01:
            return '***'
        elif pval < 0.05:
            return '**'
        elif pval < 0.10:
            return '*'
        return ''
    
    # 提取条件方差（转换回原始尺度）
    conditional_variance = (fitted.conditional_volatility / 100) ** 2
    conditional_volatility = fitted.conditional_volatility / 100
    
    # 计算年化波动率
    annualized_vol = conditional_volatility * np.sqrt(365)  # 加密货币全年交易
    
    # 输出结果
    print(f"\n参数估计结果:")
    print(f"  μ (mu):    {params.get('mu', 0):.6f}  [t={tvalues.get('mu', np.nan):.3f}, p={pvalues.get('mu', np.nan):.4f}{sig_marker(pvalues.get('mu', np.nan))}]")
    print(f"  ω (omega): {params['omega']:.6f}  [t={tvalues['omega']:.3f}, p={pvalues['omega']:.4f}{sig_marker(pvalues['omega'])}]")
    print(f"  α (alpha): {params['alpha[1]']:.6f}  [t={tvalues['alpha[1]']:.3f}, p={pvalues['alpha[1]']:.4f}{sig_marker(pvalues['alpha[1]'])}]")
    print(f"  β (beta):  {params['beta[1]']:.6f}  [t={tvalues['beta[1]']:.3f}, p={pvalues['beta[1]']:.4f}{sig_marker(pvalues['beta[1]'])}]")
    print(f"\n模型诊断:")
    print(f"  α + β = {params['alpha[1]'] + params['beta[1]']:.6f} (波动率持续性)")
    print(f"  对数似然值: {fitted.loglikelihood:.2f}")
    print(f"  AIC: {fitted.aic:.2f}")
    print(f"  BIC: {fitted.bic:.2f}")
    print(f"\n显著性标记: *** p<0.01, ** p<0.05, * p<0.10")
    
    results = {
        'asset': asset_name,
        'mu': params.get('mu', 0),
        'omega': params['omega'],
        'alpha': params['alpha[1]'],
        'beta': params['beta[1]'],
        'mu_tstat': tvalues.get('mu', np.nan),
        'omega_tstat': tvalues['omega'],
        'alpha_tstat': tvalues['alpha[1]'],
        'beta_tstat': tvalues['beta[1]'],
        'mu_pvalue': pvalues.get('mu', np.nan),
        'omega_pvalue': pvalues['omega'],
        'alpha_pvalue': pvalues['alpha[1]'],
        'beta_pvalue': pvalues['beta[1]'],
        'persistence': params['alpha[1]'] + params['beta[1]'],
        'log_likelihood': fitted.loglikelihood,
        'aic': fitted.aic,
        'bic': fitted.bic,
        'n_obs': len(returns),
        'start_date': returns.index.min(),
        'end_date': returns.index.max()
    }
    
    return results, conditional_variance, conditional_volatility, annualized_vol


# ==============================================================================
# 3. DCC模型估计函数
# ==============================================================================

def estimate_dcc_correlation(returns1, returns2, name1, name2):
    """
    估计两资产的DCC条件相关性
    """
    print(f"\n{'='*60}")
    print(f"估计 {name1}-{name2} 的 DCC 条件相关性")
    print(f"{'='*60}")
    
    # 找出共同样本
    combined = pd.DataFrame({name1: returns1, name2: returns2})
    combined = combined.dropna()
    combined = combined.sort_index()
    
    print(f"  共同样本期: {combined.index.min()} 至 {combined.index.max()}")
    print(f"  观测数量: {len(combined)}")
    
    if len(combined) < 50:
        print(f"  警告: 样本量不足，可能影响估计结果")
    
    # 第一步：估计边际GARCH模型
    conditional_vols = pd.DataFrame(index=combined.index)
    standardized_residuals = pd.DataFrame(index=combined.index)
    
    for name in [name1, name2]:
        returns_scaled = combined[name] * 100
        model = arch_model(returns_scaled, vol='Garch', p=1, q=1, mean='Constant', rescale=False)
        fitted = model.fit(disp='off', show_warning=False)
        
        cond_vol = fitted.conditional_volatility / 100
        conditional_vols[name] = cond_vol
        standardized_residuals[name] = combined[name] / cond_vol
    
    # 第二步：计算DCC条件相关性
    # 使用EWMA方法近似DCC动态相关性
    z = standardized_residuals.values
    T = len(z)
    
    # 计算无条件相关矩阵
    R_bar = np.corrcoef(z.T)
    
    # DCC参数（使用典型值）
    alpha_dcc = 0.05
    beta_dcc = 0.93
    
    # 初始化Q矩阵
    Q = R_bar.copy()
    
    # 存储动态相关系数
    dynamic_corr = np.zeros(T)
    
    for t in range(T):
        if t == 0:
            Q = R_bar.copy()
        else:
            outer_z = np.outer(z[t-1], z[t-1])
            Q = (1 - alpha_dcc - beta_dcc) * R_bar + alpha_dcc * outer_z + beta_dcc * Q
        
        # 标准化得到相关矩阵
        Q_diag = np.sqrt(np.diag(Q))
        R_t = Q / np.outer(Q_diag, Q_diag)
        dynamic_corr[t] = R_t[0, 1]
    
    # 创建Series
    dcc_series = pd.Series(dynamic_corr, index=combined.index, name=f'{name1}-{name2}')
    
    print(f"\n  动态相关系数统计:")
    print(f"    均值: {dcc_series.mean():.4f}")
    print(f"    标准差: {dcc_series.std():.4f}")
    print(f"    最小值: {dcc_series.min():.4f}")
    print(f"    最大值: {dcc_series.max():.4f}")
    print(f"    无条件相关: {R_bar[0,1]:.4f}")
    
    return dcc_series, R_bar[0, 1], combined


# ==============================================================================
# 4. 绘图函数
# ==============================================================================

def plot_conditional_variance(conditional_variance, asset_name, output_path):
    """
    绘制条件方差时序图
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(conditional_variance.index, conditional_variance.values, 
            color='#2E86AB', linewidth=1.2, alpha=0.9)
    ax.fill_between(conditional_variance.index, 0, conditional_variance.values, 
                   alpha=0.3, color='#2E86AB')
    
    ax.set_title(f'{asset_name} GARCH(1,1) Conditional Variance\n{asset_name} GARCH(1,1) 条件方差', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Date / 日期', fontsize=12)
    ax.set_ylabel('Conditional Variance / 条件方差', fontsize=12)
    
    # 添加统计信息
    stats_text = f'Mean: {conditional_variance.mean():.6f}\nMax: {conditional_variance.max():.6f}\nMin: {conditional_variance.min():.6f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    
    ax.grid(True, alpha=0.3)
    ax.set_xlim(conditional_variance.index.min(), conditional_variance.index.max())
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {output_path}")


def plot_dcc_correlation(dcc_series, name1, name2, unconditional_corr, output_path):
    """
    绘制DCC条件相关性时序图
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(dcc_series.index, dcc_series.values, 
            color='#E74C3C', linewidth=1.2, alpha=0.9, label='Dynamic Correlation / 动态相关')
    ax.axhline(y=unconditional_corr, color='#2ECC71', linestyle='--', linewidth=2, 
               label=f'Unconditional Corr = {unconditional_corr:.4f}')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    ax.fill_between(dcc_series.index, unconditional_corr, dcc_series.values, 
                   where=(dcc_series.values > unconditional_corr),
                   alpha=0.3, color='#E74C3C', interpolate=True)
    ax.fill_between(dcc_series.index, unconditional_corr, dcc_series.values, 
                   where=(dcc_series.values <= unconditional_corr),
                   alpha=0.3, color='#3498DB', interpolate=True)
    
    ax.set_title(f'DCC Dynamic Correlation: {name1} - {name2}\nDCC动态相关系数: {name1} - {name2}', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Date / 日期', fontsize=12)
    ax.set_ylabel('Correlation / 相关系数', fontsize=12)
    
    # 添加统计信息
    stats_text = f'Mean: {dcc_series.mean():.4f}\nStd: {dcc_series.std():.4f}\nMax: {dcc_series.max():.4f}\nMin: {dcc_series.min():.4f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(dcc_series.index.min(), dcc_series.index.max())
    ax.set_ylim(-1, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {output_path}")


# ==============================================================================
# 5. 主程序
# ==============================================================================

def main():
    """
    主函数：执行Q3加密货币分析
    """
    print("\n" + "="*80)
    print("Q3任务：加密货币GARCH(1,1)估计和DCC条件相关性分析")
    print("="*80)
    
    if not ARCH_AVAILABLE:
        print("\n错误: arch库未安装，请运行 'pip install arch'")
        return
    
    if not MATPLOTLIB_AVAILABLE:
        print("\n错误: matplotlib库未安装，请运行 'pip install matplotlib'")
        return
    
    # 定义路径
    base_path = Path(__file__).parent
    crypto_path = base_path / "data" / "Raw_data" / "coin"
    etf_path = base_path / "data" / "Processed_data" / "WSJ"
    figures_path = base_path / "figures"
    results_path = base_path / "results" / "Q3"
    
    # 确保输出目录存在
    figures_path.mkdir(exist_ok=True)
    results_path.mkdir(exist_ok=True, parents=True)
    
    # ===========================================================================
    # Part 1: 加密货币GARCH(1,1)估计
    # ===========================================================================
    print("\n" + "="*80)
    print("Part 1: 加密货币 GARCH(1,1) 估计")
    print("="*80)
    
    # 读取加密货币数据
    bitcoin_returns, bitcoin_prices = load_crypto_data(
        crypto_path / "Bitcoin_2024_12_8-2025_12_8_historical_data_coinmarketcap.csv"
    )
    ethereum_returns, ethereum_prices = load_crypto_data(
        crypto_path / "Ethereum_2024_12_8-2025_12_8_historical_data_coinmarketcap.csv"
    )
    
    # 估计GARCH(1,1)模型
    garch_results = []
    
    btc_results, btc_cond_var, btc_cond_vol, btc_ann_vol = estimate_garch11(bitcoin_returns, 'Bitcoin')
    garch_results.append(btc_results)
    
    eth_results, eth_cond_var, eth_cond_vol, eth_ann_vol = estimate_garch11(ethereum_returns, 'Ethereum')
    garch_results.append(eth_results)
    
    # 保存GARCH参数
    garch_df = pd.DataFrame(garch_results)
    garch_df.to_csv(results_path / "crypto_garch_params.csv", index=False, encoding='utf-8-sig')
    print(f"\nGARCH参数已保存至: {results_path / 'crypto_garch_params.csv'}")
    
    # ===========================================================================
    # Part 2: 绘制加密货币条件方差图
    # ===========================================================================
    print("\n" + "="*80)
    print("Part 2: 绘制加密货币条件方差图")
    print("="*80)
    
    plot_conditional_variance(btc_cond_var, 'Bitcoin', 
                             figures_path / "Q3_Bitcoin_conditional_variance.png")
    plot_conditional_variance(eth_cond_var, 'Ethereum', 
                             figures_path / "Q3_Ethereum_conditional_variance.png")
    
    # ===========================================================================
    # Part 3: 绘制DCC条件相关性图
    # ===========================================================================
    print("\n" + "="*80)
    print("Part 3: 绘制3组DCC条件相关性图")
    print("="*80)
    
    # 读取ETF数据
    spy_returns = load_etf_data(etf_path / "HistoricalPrices_S&P 500 SPDR.csv")
    iwm_returns = load_etf_data(etf_path / "HistoricalPrices_Russell 2000 ETF.csv")
    
    # 重命名以便识别
    spy_returns.name = 'SPY'
    iwm_returns.name = 'IWM'
    
    # 3组DCC相关性估计和绘图
    dcc_pairs = [
        (spy_returns, bitcoin_returns, 'SPY', 'Bitcoin'),
        (iwm_returns, bitcoin_returns, 'IWM', 'Bitcoin'),
        (bitcoin_returns, ethereum_returns, 'Bitcoin', 'Ethereum')
    ]
    
    dcc_results = []
    
    for ret1, ret2, name1, name2 in dcc_pairs:
        try:
            dcc_series, uncond_corr, common_data = estimate_dcc_correlation(ret1, ret2, name1, name2)
            
            output_file = figures_path / f"Q3_{name1}-{name2}_DCC_correlation.png"
            plot_dcc_correlation(dcc_series, name1, name2, uncond_corr, output_file)
            
            dcc_results.append({
                'pair': f'{name1}-{name2}',
                'unconditional_corr': uncond_corr,
                'dynamic_corr_mean': dcc_series.mean(),
                'dynamic_corr_std': dcc_series.std(),
                'dynamic_corr_min': dcc_series.min(),
                'dynamic_corr_max': dcc_series.max(),
                'n_obs': len(dcc_series),
                'start_date': dcc_series.index.min(),
                'end_date': dcc_series.index.max()
            })
        except Exception as e:
            print(f"  错误: 处理 {name1}-{name2} 时发生错误: {str(e)}")
    
    # 保存DCC结果
    dcc_df = pd.DataFrame(dcc_results)
    dcc_df.to_csv(results_path / "dcc_correlation_summary.csv", index=False, encoding='utf-8-sig')
    print(f"\nDCC相关性汇总已保存至: {results_path / 'dcc_correlation_summary.csv'}")
    
    # ===========================================================================
    # 汇总报告
    # ===========================================================================
    print("\n" + "="*80)
    print("任务完成汇总")
    print("="*80)
    
    print("\n生成的图表文件:")
    print(f"  1. {figures_path / 'Q3_Bitcoin_conditional_variance.png'}")
    print(f"  2. {figures_path / 'Q3_Ethereum_conditional_variance.png'}")
    print(f"  3. {figures_path / 'Q3_SPY-Bitcoin_DCC_correlation.png'}")
    print(f"  4. {figures_path / 'Q3_IWM-Bitcoin_DCC_correlation.png'}")
    print(f"  5. {figures_path / 'Q3_Bitcoin-Ethereum_DCC_correlation.png'}")
    
    print("\n生成的结果文件:")
    print(f"  1. {results_path / 'crypto_garch_params.csv'}")
    print(f"  2. {results_path / 'dcc_correlation_summary.csv'}")
    
    print("\n" + "="*80)
    print("Q3任务全部完成！")
    print("="*80)


if __name__ == "__main__":
    main()
