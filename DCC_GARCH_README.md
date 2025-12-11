# DCC-GARCH模型估计说明文档

## 目录
1. [DCC-GARCH模型简介](#dcc-garch模型简介)
2. [模型理论](#模型理论)
3. [实现方法](#实现方法)
4. [使用说明](#使用说明)
5. [结果解读](#结果解读)
6. [常见问题](#常见问题)

---

## DCC-GARCH模型简介

**DCC-GARCH**（Dynamic Conditional Correlation GARCH，动态条件相关GARCH）是一种多变量GARCH模型，用于估计多个资产之间的时变条件相关系数。

### 核心特点

- **时变相关性**：相关系数随时间动态变化
- **两阶段估计**：先估计边际GARCH模型，再估计DCC参数
- **计算效率高**：比完全多变量GARCH模型更容易估计
- **灵活性好**：可以处理不同资产组合

---

## 模型理论

### 1. 边际GARCH模型（第一阶段）

对每个资产 $i$，分别估计GARCH(1,1)模型：

**均值方程**：
$$r_{i,t} = \mu_i + \varepsilon_{i,t}$$

**方差方程**：
$$\sigma_{i,t}^2 = \omega_i + \alpha_i \varepsilon_{i,t-1}^2 + \beta_i \sigma_{i,t-1}^2$$

其中：
- $r_{i,t}$：资产 $i$ 在时期 $t$ 的收益率
- $\sigma_{i,t}^2$：资产 $i$ 在时期 $t$ 的条件方差
- $\omega_i, \alpha_i, \beta_i$：GARCH参数（需满足 $\alpha_i + \beta_i < 1$ 以保证平稳性）

### 2. 标准化残差

计算标准化残差：
$$z_{i,t} = \frac{\varepsilon_{i,t}}{\sigma_{i,t}}$$

### 3. DCC模型（第二阶段）

**动态相关系数矩阵**：
$$Q_t = (1 - \alpha_D - \beta_D) \bar{Q} + \alpha_D z_{t-1} z_{t-1}' + \beta_D Q_{t-1}$$

**标准化相关系数矩阵**：
$$R_t = \text{diag}(Q_t)^{-1/2} Q_t \text{diag}(Q_t)^{-1/2}$$

其中：
- $Q_t$：未标准化的相关矩阵
- $R_t$：标准化的条件相关矩阵
- $\bar{Q}$：标准化残差的样本协方差矩阵
- $\alpha_D, \beta_D$：DCC参数（需满足 $\alpha_D + \beta_D < 1$）

### 4. 两阶段估计方法

1. **第一阶段**：对每个资产分别估计边际GARCH模型
2. **第二阶段**：基于标准化残差估计DCC参数

---

## 实现方法

### 代码结构

```
dcc_garch_estimation.py
├── 数据读取模块
│   ├── load_etf_data()      # 读取ETF数据
│   └── load_crypto_data()    # 读取加密货币数据
├── 共同样本筛选模块
│   └── find_common_sample()  # 找出共同样本
├── DCC-GARCH估计模块
│   └── estimate_dcc_garch()  # 估计DCC-GARCH模型
├── 结果展示和保存模块
│   ├── display_results()     # 展示结果
│   └── save_results()         # 保存结果
└── 主程序
    ├── run_q2_analysis()     # Q2任务
    └── run_q3_analysis()       # Q3任务
```

### 估计流程

1. **数据准备**
   - 读取价格数据
   - 计算对数收益率：$r_t = \ln(P_t / P_{t-1})$
   - 筛选共同样本（确保所有资产在相同日期都有数据）

2. **第一阶段：边际GARCH估计**
   - 对每个资产分别估计GARCH(1,1)模型
   - 提取条件波动率 $\sigma_{i,t}$
   - 计算标准化残差 $z_{i,t}$

3. **第二阶段：DCC参数估计**
   - 计算标准化残差的样本相关矩阵 $\bar{Q}$
   - 估计DCC参数 $\alpha_D$ 和 $\beta_D$
   - 计算动态相关系数矩阵 $R_t$

---

## 使用说明

### 环境要求

```bash
pip install arch pandas numpy
```

### 数据准备

确保以下数据文件存在：

**ETF数据**（`data/Processed_data/WSJ/`）：
- `HistoricalPrices_S&P 500 SPDR.csv`
- `HistoricalPrices_Russell 2000 ETF.csv`
- `HistoricalPrices_MSCI Germany ETF.csv`
- `HistoricalPrices_MSCI Hong Kong ETF.csv`
- `HistoricalPrices_MSCI United Kingdom ETF.csv`

**加密货币数据**（`data/Raw_data/coin/`）：
- `Bitcoin_2024_12_8-2025_12_8_historical_data_coinmarketcap.csv`
- `Ethereum_2024_12_8-2025_12_8_historical_data_coinmarketcap.csv`

### 运行脚本

```bash
python dcc_garch_estimation.py
```

### 输出文件

结果保存在 `results/` 目录：

```
results/
├── Q2/
│   ├── SPY-IWM_marginal_garch_params.csv
│   ├── SPY-IWM_common_sample_info.csv
│   ├── EWG-EWU_marginal_garch_params.csv
│   ├── EWG-EWU_common_sample_info.csv
│   ├── SPY-EWG-EWH-EWU_marginal_garch_params.csv
│   └── SPY-EWG-EWH-EWU_common_sample_info.csv
└── Q3/
    ├── SPY-Bitcoin_marginal_garch_params.csv
    ├── SPY-Bitcoin_common_sample_info.csv
    ├── IWM-Bitcoin_marginal_garch_params.csv
    ├── IWM-Bitcoin_common_sample_info.csv
    ├── Bitcoin-Ethereum_marginal_garch_params.csv
    └── Bitcoin-Ethereum_common_sample_info.csv
```

---

## 结果解读

### 1. 边际GARCH参数文件

**文件格式**：`*_marginal_garch_params.csv`

**列说明**：
- `资产`：资产名称
- `omega`：GARCH模型的常数项 $\omega$
- `omega_tstat`：$\omega$ 的t统计量
- `omega_pvalue`：$\omega$ 的p值
- `alpha`：ARCH项系数 $\alpha$
- `alpha_tstat`：$\alpha$ 的t统计量
- `alpha_pvalue`：$\alpha$ 的p值
- `beta`：GARCH项系数 $\beta$
- `beta_tstat`：$\beta$ 的t统计量
- `beta_pvalue`：$\beta$ 的p值
- `mu`：均值项 $\mu$
- `log_likelihood`：对数似然值
- `AIC`：Akaike信息准则
- `BIC`：Bayesian信息准则

**显著性标记**：
- `***`：p < 0.01（高度显著）
- `**`：p < 0.05（显著）
- `*`：p < 0.10（弱显著）

### 2. 共同样本信息文件

**文件格式**：`*_common_sample_info.csv`

**列说明**：
- `组合名称`：资产组合名称
- `开始日期`：共同样本的开始日期
- `结束日期`：共同样本的结束日期
- `观测数量`：共同样本的观测数量
- `资产`：包含的资产列表

### 3. 参数合理性检查

**GARCH参数**：
- $\omega > 0$：常数项应为正
- $\alpha \geq 0, \beta \geq 0$：ARCH和GARCH项系数应为非负
- $\alpha + \beta < 1$：保证平稳性
- $\alpha + \beta$ 接近1：表示接近IGARCH特征

**显著性检验**：
- 所有参数应在5%或10%水平显著
- t统计量绝对值应大于1.96（5%水平）或1.645（10%水平）

**DCC参数**：
- $\alpha_D \geq 0, \beta_D \geq 0$：DCC参数应为非负
- $\alpha_D + \beta_D < 1$：保证平稳性
- $\alpha_D + \beta_D$ 接近1：表示相关性高度持续

---

## 常见问题

### 1. 参数估计不收敛

**可能原因**：
- 样本量不足
- 数据中存在异常值
- 初始值设置不当

**解决方法**：
- 增加样本量
- 检查并处理异常值
- 调整优化算法参数

### 2. 参数不显著

**可能原因**：
- 样本量太小
- 数据波动性不足
- 模型设定不当

**解决方法**：
- 使用更长的样本期
- 检查数据质量
- 尝试其他GARCH变体（如EGARCH、GJR-GARCH）

### 3. 共同样本为空

**可能原因**：
- 不同资产的数据日期范围没有重叠
- 数据中存在大量缺失值

**解决方法**：
- 检查数据的时间范围
- 确认数据预处理是否正确
- 使用更灵活的数据对齐方法

### 4. Ethereum参数异常

**现象**：$\omega$ 值异常高（如7.7470）

**可能原因**：
- 数据质量问题
- 样本量不足
- 模型不适合该资产

**解决方法**：
- 检查Ethereum数据质量
- 增加样本量
- 尝试其他GARCH模型

### 5. DCC参数估计问题

**当前实现**：使用简化方法估计DCC参数

**完整实现**：应使用最大似然估计或专门的DCC估计器

**建议**：
- 使用 `arch.multivariate.DCC` 模块
- 或使用R语言的 `rmgarch` 包

---

## 任务要求

### Q2任务

1. **资产对估计**：
   - (a) S&P 500 SPDR ETF 和 Russell 2000 iShares ETF
   - (b) iShares MSCI Germany 和 iShares MSCI UK

2. **4资产组合估计**：
   - S&P 500 SPDR ETF, iShares MSCI Germany, iShares MSCI Hong Kong, iShares MSCI UK

### Q3任务

1. S&P 500 SPDR ETF 和 Bitcoin 的GARCH-DCC模型
2. Russell 2000 iShares ETF 和 Bitcoin 的GARCH-DCC模型
3. Bitcoin 和 Ethereum 的GARCH-DCC模型

所有估计都使用共同样本。

---

## 技术细节

### 对数收益率计算

$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

### GARCH(1,1)模型

$$\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

### DCC模型

$$Q_t = (1 - \alpha_D - \beta_D) \bar{Q} + \alpha_D z_{t-1} z_{t-1}' + \beta_D Q_{t-1}$$

$$R_t = \text{diag}(Q_t)^{-1/2} Q_t \text{diag}(Q_t)^{-1/2}$$

### 数值稳定性

- 收益率乘以100以提高数值稳定性
- 使用标准化残差避免数值问题
- 检查参数约束条件

---

## 参考文献

1. Engle, R. (2002). Dynamic conditional correlation: A simple class of multivariate generalized autoregressive conditional heteroskedasticity models. *Journal of Business & Economic Statistics*, 20(3), 339-350.

2. Engle, R., & Sheppard, K. (2001). Theoretical and empirical properties of dynamic conditional correlation multivariate GARCH. *NBER Working Paper*.

3. Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307-327.

---

## 联系信息

如有问题，请联系成员4。

---

**最后更新**：2025年12月

