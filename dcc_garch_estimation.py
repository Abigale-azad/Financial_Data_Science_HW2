"""
DCC-GARCH模型估计
用于估计多变量动态条件相关GARCH模型
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("警告: arch库未安装，请运行: pip install arch")

try:
    from scipy.optimize import minimize
    from scipy.linalg import inv
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("警告: scipy库未安装，请运行: pip install scipy")

def load_etf_data(file_path):
    """读取ETF数据并计算对数收益率"""
    print(f"\n正在读取: {file_path.name}")
    df = pd.read_csv(file_path)
    
    if 'Date' not in df.columns:
        raise ValueError(f"文件 {file_path.name} 中未找到Date列")
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    if 'Close' not in df.columns:
        raise ValueError(f"文件 {file_path.name} 中未找到Close列")
    
    prices = df['Close']
    returns = np.log(prices / prices.shift(1)).dropna()
    returns.name = file_path.stem.replace('HistoricalPrices_', '')
    print(f"  数据范围: {returns.index.min()} 至 {returns.index.max()}")
    print(f"  观测数量: {len(returns)}")
    return returns


def load_crypto_data(file_path):
    """读取加密货币数据并计算对数收益率"""
    print(f"\n正在读取: {file_path.name}")
    try:
        df = pd.read_csv(file_path, sep=';')
    except Exception as e:
        print(f"  错误: 无法读取文件: {str(e)}")
        raise
    
    date_col = 'timeOpen' if 'timeOpen' in df.columns else 'timestamp'
    if date_col not in df.columns:
        raise ValueError(f"文件 {file_path.name} 中未找到日期列")
    
    date_str = df[date_col].astype(str).str.replace('"', '')
    df['Date'] = pd.to_datetime(date_str, errors='coerce', format='ISO8601')
    
    if df['Date'].isna().any():
        date_str_clean = date_str.str.extract(r'(\d{4}-\d{2}-\d{2})')[0]
        df['Date'] = pd.to_datetime(date_str_clean, errors='coerce')
    
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna(subset=['Date'])
    
    if len(df) == 0:
        raise ValueError(f"文件 {file_path.name} 中没有有效的日期数据")
    
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    if 'close' not in df.columns:
        raise ValueError(f"文件 {file_path.name} 中未找到close列")
    
    prices = pd.to_numeric(df['close'].astype(str).str.replace('"', ''), errors='coerce').dropna()
    if len(prices) == 0:
        raise ValueError(f"文件 {file_path.name} 中没有有效的价格数据")
    
    returns = np.log(prices / prices.shift(1)).dropna()
    if len(returns) == 0:
        raise ValueError(f"文件 {file_path.name} 无法计算收益率")
    
    returns = returns[np.abs(returns) <= 1.0]
    crypto_name = 'Bitcoin' if 'Bitcoin' in file_path.name else 'Ethereum'
    returns.name = crypto_name
    
    print(f"  数据范围: {returns.index.min()} 至 {returns.index.max()}")
    print(f"  观测数量: {len(returns)}")
    print(f"  收益率统计: 均值={returns.mean():.6f}, 标准差={returns.std():.6f}")
    return returns


def find_common_sample(*returns_series, names=None):
    """找出多个时间序列的共同样本"""
    if names is None:
        names = [f"Asset_{i+1}" for i in range(len(returns_series))]
    
    if len(returns_series) == 0:
        raise ValueError("至少需要一个收益率序列")
    if len(returns_series) != len(names):
        raise ValueError("收益率序列数量与名称数量不匹配")
    
    for i, ret in enumerate(returns_series):
        if len(ret) == 0:
            raise ValueError(f"序列 {names[i]} 为空")
        if not isinstance(ret, pd.Series):
            raise ValueError(f"序列 {names[i]} 不是pandas Series")
    
    combined = pd.DataFrame()
    for i, ret in enumerate(returns_series):
        if not isinstance(ret.index, pd.DatetimeIndex):
            ret.index = pd.to_datetime(ret.index)
        ret.index = pd.to_datetime(ret.index.date)
        combined[names[i]] = ret
    
    common_data = combined.dropna()
    
    if len(common_data) > 0:
        common_data.index = pd.to_datetime(common_data.index)
        common_data = common_data.sort_index()
    
    if len(common_data) == 0:
        print(f"\n警告: 资产 {', '.join(names)} 的共同样本为空")
        for i, ret in enumerate(returns_series):
            print(f"  {names[i]}: {len(ret)} 个观测值, 日期范围 {ret.index.min()} 至 {ret.index.max()}")
    
    info = {
        'start_date': common_data.index.min() if len(common_data) > 0 else None,
        'end_date': common_data.index.max() if len(common_data) > 0 else None,
        'n_obs': len(common_data),
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


def estimate_dcc_garch(returns_data, asset_names):
    """估计DCC-GARCH模型"""
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
        'dcc_params': None
    }
    
    print(f"\n第一阶段：估计边际GARCH(1,1)模型")
    print("-" * 60)
    
    standardized_residuals = pd.DataFrame(index=returns_data.index)
    
    for asset in asset_names:
        print(f"\n估计 {asset} 的GARCH(1,1)模型...")
        returns = returns_data[asset].dropna()
        
        # 数据验证
        if len(returns) == 0:
            raise ValueError(f"资产 {asset} 的数据为空，无法估计GARCH模型")
        
        if len(returns) < 100:
            print(f"  警告: 资产 {asset} 只有 {len(returns)} 个观测值，可能不足以估计GARCH模型")
        
        if returns.std() == 0 or np.isnan(returns.std()):
            raise ValueError(f"资产 {asset} 的收益率标准差为0或NaN，无法估计GARCH模型")
        
        if np.any(np.abs(returns) > 1.0):
            print(f"  警告: 资产 {asset} 存在绝对值大于1的收益率")
        
        try:
            model = arch_model(returns * 100, vol='Garch', p=1, q=1, rescale=False)
            fitted = model.fit(disp='off', show_warning=False)
        except Exception as e:
            print(f"  错误: 无法估计 {asset} 的GARCH模型")
            print(f"  错误信息: {str(e)}")
            print(f"  数据统计: 均值={returns.mean():.6f}, 标准差={returns.std():.6f}, 观测数={len(returns)}")
            raise
        
        params = fitted.params
        tvalues = fitted.tvalues
        pvalues = fitted.pvalues
        
        def significance_marker(pval):
            if pd.isna(pval) or pval >= 1:
                return ''
            if pval < 0.01:
                return '***'
            elif pval < 0.05:
                return '**'
            elif pval < 0.10:
                return '*'
            return ''
        
        def safe_get(series, key, default=np.nan):
            try:
                if key in series.index:
                    return series[key]
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
        
        conditional_vol = fitted.conditional_volatility / 100
        standardized_residuals[asset] = returns / conditional_vol
        
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
    
    print(f"\n第二阶段：估计DCC参数")
    print("-" * 60)
    
    standardized_residuals_clean = standardized_residuals.dropna()
    z = standardized_residuals_clean.values
    T, n = z.shape
    
    R_bar = np.corrcoef(z.T)
    
    print("\n使用两步估计法估计DCC参数")
    
    if n_assets == 2:
        avg_corr = R_bar[0, 1]
    else:
        mask = ~np.eye(n_assets, dtype=bool)
        avg_corr = np.mean(R_bar[mask])
    
    def dcc_loglikelihood(params, z, R_bar):
        """DCC模型的对数似然函数"""
        alpha_D, beta_D = params
        
        if alpha_D < 0 or beta_D < 0 or alpha_D + beta_D >= 1:
            return 1e10
        
        T, n = z.shape
        Q = np.zeros((T, n, n))
        Q[0] = R_bar.copy()
        
        for t in range(1, T):
            Q[t] = (1 - alpha_D - beta_D) * R_bar + \
                   alpha_D * np.outer(z[t-1], z[t-1]) + \
                   beta_D * Q[t-1]
        
        loglik = 0.0
        for t in range(T):
            Qt = Q[t]
            try:
                diag_Qt = np.diag(Qt)
                diag_Qt = np.maximum(diag_Qt, 1e-8)
                diag_Qt_inv_sqrt = 1.0 / np.sqrt(diag_Qt)
                D_inv = np.diag(diag_Qt_inv_sqrt)
                Rt = D_inv @ Qt @ D_inv
                Rt = (Rt + Rt.T) / 2
                
                eigenvals = np.linalg.eigvals(Rt)
                if np.any(eigenvals <= 0):
                    return 1e10
                
                try:
                    det_Rt = np.linalg.det(Rt)
                    if det_Rt <= 0:
                        return 1e10
                    
                    inv_Rt = np.linalg.inv(Rt)
                    zt = z[t]
                    loglik += 0.5 * (np.log(det_Rt) + zt @ inv_Rt @ zt - zt @ zt)
                except np.linalg.LinAlgError:
                    return 1e10
            except Exception:
                return 1e10
        
        return loglik
    
    if SCIPY_AVAILABLE:
        print("\n开始最大似然估计...")
        
        initial_params = [0.02, 0.95]
        bounds = [(1e-6, 0.5), (1e-6, 0.999)]
        constraints = {
            'type': 'ineq',
            'fun': lambda x: 1 - x[0] - x[1] - 1e-6
        }
        
        try:
            result = minimize(
                dcc_loglikelihood,
                initial_params,
                args=(z, R_bar),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-6, 'disp': False}
            )
            
            if result.success:
                alpha_D_est = result.x[0]
                beta_D_est = result.x[1]
                
                from scipy.linalg import inv as inv_matrix
                
                try:
                    def loglik_wrapper(p):
                        return dcc_loglikelihood(p, z, R_bar)
                    
                    eps = 1e-5
                    hessian = np.zeros((2, 2))
                    f_0 = result.fun
                    
                    for i in range(2):
                        params_plus = result.x.copy()
                        params_plus[i] += eps
                        f_plus = loglik_wrapper(params_plus)
                        
                        params_minus = result.x.copy()
                        params_minus[i] -= eps
                        f_minus = loglik_wrapper(params_minus)
                        
                        hessian[i, i] = (f_plus - 2 * f_0 + f_minus) / (eps * eps)
                    
                    for i in range(2):
                        for j in range(i + 1, 2):
                            params_pp = result.x.copy()
                            params_pp[i] += eps
                            params_pp[j] += eps
                            f_pp = loglik_wrapper(params_pp)
                            
                            params_pm = result.x.copy()
                            params_pm[i] += eps
                            params_pm[j] -= eps
                            f_pm = loglik_wrapper(params_pm)
                            
                            params_mp = result.x.copy()
                            params_mp[i] -= eps
                            params_mp[j] += eps
                            f_mp = loglik_wrapper(params_mp)
                            
                            params_mm = result.x.copy()
                            params_mm[i] -= eps
                            params_mm[j] -= eps
                            f_mm = loglik_wrapper(params_mm)
                            
                            hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps * eps)
                            hessian[j, i] = hessian[i, j]
                    
                    eigenvals = np.linalg.eigvals(hessian)
                    if np.any(eigenvals <= 0):
                        min_eigenval = np.min(eigenvals)
                        if min_eigenval <= 0:
                            hessian += np.eye(2) * (abs(min_eigenval) + 1e-6)
                    
                    try:
                        cov_matrix = inv_matrix(hessian)
                        if np.any(np.diag(cov_matrix) <= 0):
                            raise ValueError("协方差矩阵的对角线元素必须为正")
                        
                        std_errors = np.sqrt(np.diag(cov_matrix))
                        alpha_D_se = std_errors[0]
                        beta_D_se = std_errors[1]
                        
                        alpha_D_tstat = alpha_D_est / alpha_D_se if alpha_D_se > 0 else np.nan
                        beta_D_tstat = beta_D_est / beta_D_se if beta_D_se > 0 else np.nan
                        
                        from scipy.stats import norm
                        alpha_D_pvalue = 2 * (1 - norm.cdf(abs(alpha_D_tstat))) if not np.isnan(alpha_D_tstat) else np.nan
                        beta_D_pvalue = 2 * (1 - norm.cdf(abs(beta_D_tstat))) if not np.isnan(beta_D_tstat) else np.nan
                    except Exception:
                        alpha_D_se = beta_D_se = alpha_D_tstat = beta_D_tstat = np.nan
                        alpha_D_pvalue = beta_D_pvalue = np.nan
                except Exception:
                    alpha_D_se = beta_D_se = alpha_D_tstat = beta_D_tstat = np.nan
                    alpha_D_pvalue = beta_D_pvalue = np.nan
                
                print(f"  优化成功，迭代次数: {result.nit}")
                
                results['dcc_params'] = {
                    'alpha_D': alpha_D_est,
                    'beta_D': beta_D_est,
                    'alpha_D_se': alpha_D_se,
                    'beta_D_se': beta_D_se,
                    'alpha_D_tstat': alpha_D_tstat,
                    'beta_D_tstat': beta_D_tstat,
                    'alpha_D_pvalue': alpha_D_pvalue,
                    'beta_D_pvalue': beta_D_pvalue,
                    'average_correlation': avg_corr,
                    'correlation_matrix': R_bar,
                    'log_likelihood': -result.fun,
                    'note': 'DCC参数通过最大似然估计得到'
                }
                
                print(f"\nDCC参数估计结果:")
                print(f"  α_D: {alpha_D_est:.6f}")
                if not np.isnan(alpha_D_se):
                    print(f"    (标准误: {alpha_D_se:.6f}, t={alpha_D_tstat:.3f}, p={alpha_D_pvalue:.4f})")
                print(f"  β_D: {beta_D_est:.6f}")
                if not np.isnan(beta_D_se):
                    print(f"    (标准误: {beta_D_se:.6f}, t={beta_D_tstat:.3f}, p={beta_D_pvalue:.4f})")
                print(f"  α_D + β_D: {alpha_D_est + beta_D_est:.6f}")
                print(f"  对数似然值: {-result.fun:.2f}")
                
            else:
                print(f"  警告: 优化未成功，消息: {result.message}")
                raise ValueError("优化失败")
                
        except Exception as e:
            print(f"  警告: DCC估计失败: {str(e)}")
            alpha_D_est = 0.02
            beta_D_est = 0.95
            
            results['dcc_params'] = {
                'alpha_D': alpha_D_est,
                'beta_D': beta_D_est,
                'alpha_D_se': np.nan,
                'beta_D_se': np.nan,
                'alpha_D_tstat': np.nan,
                'beta_D_tstat': np.nan,
                'alpha_D_pvalue': np.nan,
                'beta_D_pvalue': np.nan,
                'average_correlation': avg_corr,
                'correlation_matrix': R_bar,
                'log_likelihood': np.nan,
                'note': 'DCC参数估计失败，使用默认值'
            }
    else:
        raise ImportError("完整DCC估计需要scipy库，请运行: pip install scipy")
    
    print(f"\n  平均相关系数: {avg_corr:.4f}")
    print(f"  无条件相关矩阵:")
    corr_df = pd.DataFrame(R_bar, index=asset_names, columns=asset_names)
    print(corr_df.round(4))
    
    return results


def display_results(results, common_sample_info, combination_name):
    """展示估计结果"""
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
        def sig_marker(pval):
            if pval < 0.01: return '***'
            if pval < 0.05: return '**'
            if pval < 0.10: return '*'
            return ''
        
        omega_sig = sig_marker(params.get('omega_pvalue', 1))
        alpha_sig = sig_marker(params.get('alpha_pvalue', 1))
        beta_sig = sig_marker(params.get('beta_pvalue', 1))
        
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
        
        alpha_D = dcc_params.get('alpha_D', np.nan)
        beta_D = dcc_params.get('beta_D', np.nan)
        alpha_D_se = dcc_params.get('alpha_D_se', np.nan)
        beta_D_se = dcc_params.get('beta_D_se', np.nan)
        alpha_D_tstat = dcc_params.get('alpha_D_tstat', np.nan)
        beta_D_tstat = dcc_params.get('beta_D_tstat', np.nan)
        alpha_D_pvalue = dcc_params.get('alpha_D_pvalue', np.nan)
        beta_D_pvalue = dcc_params.get('beta_D_pvalue', np.nan)
        log_likelihood = dcc_params.get('log_likelihood', np.nan)
        
        def sig_marker(pval):
            if np.isnan(pval): return ''
            if pval < 0.01: return '***'
            if pval < 0.05: return '**'
            if pval < 0.10: return '*'
            return ''
        
        alpha_sig = sig_marker(alpha_D_pvalue)
        beta_sig = sig_marker(beta_D_pvalue)
        
        print(f"  α_D: {alpha_D:.6f}{alpha_sig}")
        if not np.isnan(alpha_D_se):
            print(f"    (标准误: {alpha_D_se:.6f}, t统计量: {alpha_D_tstat:.3f}, p值: {alpha_D_pvalue:.4f})")
        print(f"  β_D: {beta_D:.6f}{beta_sig}")
        if not np.isnan(beta_D_se):
            print(f"    (标准误: {beta_D_se:.6f}, t统计量: {beta_D_tstat:.3f}, p值: {beta_D_pvalue:.4f})")
        if not np.isnan(alpha_D) and not np.isnan(beta_D):
            print(f"  α_D + β_D: {alpha_D + beta_D:.6f}")
        if not np.isnan(log_likelihood):
            print(f"  对数似然值: {log_likelihood:.2f}")
        print(f"  平均相关系数: {dcc_params.get('average_correlation', np.nan):.4f}")
        print(f"  注意: {dcc_params.get('note', '')}")
        print("\n显著性标记: *** p<0.01, ** p<0.05, * p<0.10")


def save_results(results, common_sample_info, combination_name, output_dir):
    """保存结果到CSV文件"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
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
    
    if results['dcc_params']:
        dcc_params = results['dcc_params']
        dcc_data = {
            '参数': ['alpha_D', 'beta_D'],
            '估计值': [
                dcc_params.get('alpha_D', np.nan),
                dcc_params.get('beta_D', np.nan)
            ],
            '标准误': [
                dcc_params.get('alpha_D_se', np.nan),
                dcc_params.get('beta_D_se', np.nan)
            ],
            't统计量': [
                dcc_params.get('alpha_D_tstat', np.nan),
                dcc_params.get('beta_D_tstat', np.nan)
            ],
            'p值': [
                dcc_params.get('alpha_D_pvalue', np.nan),
                dcc_params.get('beta_D_pvalue', np.nan)
            ]
        }
        
        if 'log_likelihood' in dcc_params:
            dcc_data['对数似然值'] = [dcc_params['log_likelihood'], np.nan]
        if 'average_correlation' in dcc_params:
            dcc_data['平均相关系数'] = [dcc_params['average_correlation'], np.nan]
        
        dcc_df = pd.DataFrame(dcc_data)
        dcc_file = output_dir / f"{combination_name}_dcc_params.csv"
        dcc_df.to_csv(dcc_file, index=False, encoding='utf-8-sig')
        print(f"DCC参数已保存至: {dcc_file}")
        
        if 'correlation_matrix' in dcc_params:
            corr_matrix = dcc_params['correlation_matrix']
            corr_df = pd.DataFrame(
                corr_matrix,
                index=results['assets'],
                columns=results['assets']
            )
            corr_file = output_dir / f"{combination_name}_unconditional_correlation_matrix.csv"
            corr_df.to_csv(corr_file, index=True, encoding='utf-8-sig')
            print(f"无条件相关矩阵已保存至: {corr_file}")


def run_q2_analysis():
    """执行Q2任务：ETF资产组合的DCC-GARCH估计"""
    print("\n" + "="*80)
    print("Q2任务：ETF资产组合的DCC-GARCH估计")
    print("="*80)
    
    base_path = Path(__file__).parent
    data_path = base_path / "data" / "Processed_data" / "WSJ"
    output_path = base_path / "results" / "Q2"
    
    print("\n读取ETF数据...")
    spy = load_etf_data(data_path / "HistoricalPrices_S&P 500 SPDR.csv")
    iwm = load_etf_data(data_path / "HistoricalPrices_Russell 2000 ETF.csv")
    ewg = load_etf_data(data_path / "HistoricalPrices_MSCI Germany ETF.csv")
    ewh = load_etf_data(data_path / "HistoricalPrices_MSCI Hong Kong ETF.csv")
    ewu = load_etf_data(data_path / "HistoricalPrices_MSCI United Kingdom ETF.csv")
    
    print("\n" + "="*80)
    print("Q2.1: 估计2个资产对的DCC-GARCH模型")
    print("="*80)
    
    pairs = [
        (spy, iwm, "SPY-IWM"),
        (ewg, ewu, "EWG-EWU")
    ]
    
    q2_results = {}
    
    for asset1, asset2, pair_name in pairs:
        print(f"\n处理资产对: {pair_name}")
        try:
            common_data, sample_info = find_common_sample(
                asset1, asset2,
                names=[asset1.name, asset2.name]
            )
            
            if common_data.empty:
                print(f"  错误: {pair_name} 的共同样本为空，跳过该组合")
                continue
            
            if len(common_data) < 50:
                print(f"  警告: {pair_name} 的共同样本只有 {len(common_data)} 个观测值")
            
            results = estimate_dcc_garch(common_data, [asset1.name, asset2.name])
        except Exception as e:
            print(f"  错误: 处理 {pair_name} 时发生错误: {str(e)}")
            continue
        
        display_results(results, sample_info, pair_name)
        save_results(results, sample_info, pair_name, output_path)
        
        q2_results[pair_name] = {
            'results': results,
            'sample_info': sample_info
        }
    
    print("\n" + "="*80)
    print("Q2.2: 估计4个ETF资产的DCC-GARCH模型")
    print("="*80)
    
    try:
        common_data_4, sample_info_4 = find_common_sample(
            spy, ewg, ewh, ewu,
            names=['SPY', 'EWG', 'EWH', 'EWU']
        )
        
        if common_data_4.empty:
            print("  错误: 4资产组合的共同样本为空，跳过该组合")
            results_4 = None
            sample_info_4 = None
        elif len(common_data_4) < 50:
            print(f"  警告: 4资产组合的共同样本只有 {len(common_data_4)} 个观测值")
            results_4 = None
            sample_info_4 = None
        else:
            results_4 = estimate_dcc_garch(common_data_4, ['SPY', 'EWG', 'EWH', 'EWU'])
    except Exception as e:
        print(f"  错误: 处理4资产组合时发生错误: {str(e)}")
        results_4 = None
        sample_info_4 = None
    
    if results_4 is not None:
        display_results(results_4, sample_info_4, "SPY-EWG-EWH-EWU")
        save_results(results_4, sample_info_4, "SPY-EWG-EWH-EWU", output_path)
    
    print("\n" + "="*80)
    print("Q2任务完成！")
    print("="*80)
    
    return q2_results, results_4, sample_info_4


def run_q3_analysis():
    """执行Q3任务：加密货币与ETF的DCC-GARCH估计"""
    print("\n" + "="*80)
    print("Q3任务：加密货币与ETF的DCC-GARCH估计")
    print("="*80)
    
    base_path = Path(__file__).parent
    etf_path = base_path / "data" / "Processed_data" / "WSJ"
    crypto_path = base_path / "data" / "Raw_data" / "coin"
    output_path = base_path / "results" / "Q3"
    
    print("\n读取ETF数据...")
    spy = load_etf_data(etf_path / "HistoricalPrices_S&P 500 SPDR.csv")
    iwm = load_etf_data(etf_path / "HistoricalPrices_Russell 2000 ETF.csv")
    
    print("\n读取加密货币数据...")
    bitcoin = load_crypto_data(crypto_path / "Bitcoin_2024_12_8-2025_12_8_historical_data_coinmarketcap.csv")
    ethereum = load_crypto_data(crypto_path / "Ethereum_2024_12_8-2025_12_8_historical_data_coinmarketcap.csv")
    
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
            common_data, sample_info = find_common_sample(
                asset1, asset2,
                names=[asset1.name, asset2.name]
            )
            
            if common_data.empty:
                print(f"  错误: {combo_name} 的共同样本为空，跳过该组合")
                continue
            
            if len(common_data) < 50:
                print(f"  警告: {combo_name} 的共同样本只有 {len(common_data)} 个观测值")
            
            results = estimate_dcc_garch(common_data, [asset1.name, asset2.name])
        except Exception as e:
            print(f"  错误: 处理 {combo_name} 时发生错误: {str(e)}")
            continue
        
        display_results(results, sample_info, combo_name)
        save_results(results, sample_info, combo_name, output_path)
        
        q3_results[combo_name] = {
            'results': results,
            'sample_info': sample_info
        }
    
    print("\n" + "="*80)
    print("Q3任务完成！")
    print("="*80)
    
    return q3_results


def main():
    """主函数：执行所有DCC-GARCH估计任务"""
    print("\n" + "="*80)
    print("DCC-GARCH模型估计")
    print("="*80)
    
    if not ARCH_AVAILABLE:
        print("\n错误: arch库未安装，请运行: pip install arch")
        return
    
    try:
        run_q2_analysis()
        run_q3_analysis()
        
        print("\n" + "="*80)
        print("所有任务完成！")
        print("="*80)
        print("\n结果文件已保存至 results/ 目录")
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

