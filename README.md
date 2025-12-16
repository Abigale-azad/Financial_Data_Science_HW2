# 金融数据科学作业2 - GARCH与DCC-GARCH模型分析

## 📋 项目概述

本项目对ETF和加密货币资产进行波动率建模与动态相关性分析，主要内容包括：

- **Q2**: ETF资产组合的GARCH(1,1)和DCC-GARCH模型估计
- **Q3**: 加密货币（Bitcoin、Ethereum）与传统ETF的GARCH(1,1)和DCC动态相关性分析

## 📁 项目结构

```
Financial_Data_Science_HW2/
├── README.md                           # 项目说明文档
├── complete_analysis.py                # 完整分析脚本（Q2 + Q3综合）
├── q3_crypto_analysis.py               # Q3加密货币专项分析脚本
├── dcc_garch_estimation.py             # DCC-GARCH模型估计核心脚本
├── process_wsj_data.py                 # WSJ数据预处理脚本
├── COMPREHENSIVE_ANALYSIS_REPORT.txt   # 综合分析报告
│
├── data/
│   ├── Processed_data/
│   │   └── WSJ/                        # 处理后的ETF数据
│   │       ├── HistoricalPrices_MSCI Germany ETF.csv
│   │       ├── HistoricalPrices_MSCI Hong Kong ETF.csv
│   │       ├── HistoricalPrices_MSCI United Kingdom ETF.csv
│   │       ├── HistoricalPrices_Russell 2000 ETF.csv
│   │       └── HistoricalPrices_S&P 500 SPDR.csv
│   └── Raw_data/
│       ├── coin/                       # 加密货币原始数据
│       │   ├── Bitcoin_*.csv
│       │   └── Ethereum_*.csv
│       ├── vlab/                       # V-Lab GARCH模型数据
│       └── WSJ/                        # WSJ原始数据
│
├── figures/                            # 生成的图表
│   ├── Q3_Bitcoin_conditional_variance.png
│   ├── Q3_Ethereum_conditional_variance.png
│   ├── Q3_SPY-Bitcoin_DCC_correlation.png
│   ├── Q3_IWM-Bitcoin_DCC_correlation.png
│   ├── Q3_Bitcoin-Ethereum_DCC_correlation.png
│   └── ...
│
└── results/
    ├── Q2/                             # Q2 ETF分析结果
    │   ├── EWG-EWU_marginal_garch_params.csv
    │   ├── SPY-EWG-EWH-EWU_marginal_garch_params.csv
    │   ├── SPY-IWM_marginal_garch_params.csv
    │   └── *_common_sample_info.csv
    └── Q3/                             # Q3 加密货币分析结果
        ├── crypto_garch_params.csv
        ├── dcc_correlation_summary.csv
        └── *_marginal_garch_params.csv
```

## 🔧 环境依赖

### Python版本
- Python >= 3.8

### 依赖库
```bash
pip install pandas numpy matplotlib arch
```

| 库名 | 用途 |
|------|------|
| `pandas` | 数据处理与分析 |
| `numpy` | 数值计算 |
| `matplotlib` | 图表绘制 |
| `arch` | GARCH模型估计 |

## 🚀 使用方法

### 1. 数据预处理
```bash
python process_wsj_data.py
```
- 清洗WSJ原始数据中的日期格式问题
- 输出处理后的CSV文件到 `data/Processed_data/WSJ/`

### 2. 完整分析（Q2 + Q3）
```bash
python complete_analysis.py
```
- 估计所有资产组合的边际GARCH(1,1)模型
- 计算DCC动态条件相关性
- 生成条件方差和动态相关性图表
- 输出综合分析报告

### 3. Q3加密货币专项分析
```bash
python q3_crypto_analysis.py
```
- 估计Bitcoin和Ethereum的GARCH(1,1)模型
- 计算加密货币条件方差
- 计算SPY-Bitcoin、IWM-Bitcoin、Bitcoin-Ethereum的DCC相关性
- 生成专项图表

### 4. DCC-GARCH核心估计
```bash
python dcc_garch_estimation.py
```
- 提供DCC-GARCH模型的核心估计功能

## 📊 主要结果

### Q2: ETF资产组合GARCH(1,1)参数

| 资产组合 | 资产 | ω (omega) | α (alpha) | β (beta) | 持久性 (α+β) |
|---------|------|-----------|-----------|----------|-------------|
| EWG-EWU | MSCI Germany ETF | 0.0334 | 0.0857 | 0.9019 | 0.9876 |
| EWG-EWU | MSCI UK ETF | 0.0387 | 0.1070 | 0.8744 | 0.9814 |
| SPY-IWM | S&P 500 SPDR | 0.0263 | 0.1245 | 0.8554 | 0.9799 |
| SPY-IWM | Russell 2000 ETF | 0.0410 | 0.0893 | 0.8902 | 0.9795 |

### Q3: 加密货币GARCH(1,1)参数

| 资产 | ω (omega) | α (alpha) | β (beta) | 持久性 (α+β) | 样本期 |
|------|-----------|-----------|----------|-------------|--------|
| Bitcoin | 0.1699 | 0.0513 | 0.9141 | 0.9654 | 2024-11-04 ~ 2025-12-06 |
| Ethereum | 7.7470 | 0.0758 | 0.4295 | 0.5052 | 2024-11-04 ~ 2025-12-06 |

### Q3: DCC动态相关性分析

| 资产对 | 无条件相关 | 动态相关均值 | 动态相关标准差 | 最小值 | 最大值 |
|--------|-----------|-------------|---------------|--------|--------|
| SPY-Bitcoin | -0.0813 | -0.0820 | 0.1013 | -0.3222 | 0.2175 |
| IWM-Bitcoin | -0.0422 | -0.0378 | 0.1086 | -0.2925 | 0.2342 |
| Bitcoin-Ethereum | **0.8138** | **0.8055** | 0.0732 | 0.6046 | 0.9335 |

### 共同样本信息

| 资产组合 | 样本期 | 观测数 |
|---------|--------|--------|
| EWG-EWU | 1996-03-18 ~ 2025-12-04 | 7,406 |
| SPY-EWG-EWH-EWU | 1996-03-18 ~ 2025-12-04 | 7,404 |
| SPY-IWM | 2000-05-26 ~ 2025-12-04 | 6,420 |
| Bitcoin-Ethereum | 2024-11-04 ~ 2025-12-06 | 398 |
| SPY-Bitcoin | 2024-11-04 ~ 2025-12-04 | 272 |
| IWM-Bitcoin | 2024-11-04 ~ 2025-12-04 | 272 |

## 📈 生成的图表

### Q3 条件方差图
- `Q3_Bitcoin_conditional_variance.png` - Bitcoin GARCH(1,1)条件方差时序图
- `Q3_Ethereum_conditional_variance.png` - Ethereum GARCH(1,1)条件方差时序图

### Q3 DCC动态相关性图
- `Q3_SPY-Bitcoin_DCC_correlation.png` - S&P 500与Bitcoin的动态相关性
- `Q3_IWM-Bitcoin_DCC_correlation.png` - Russell 2000与Bitcoin的动态相关性
- `Q3_Bitcoin-Ethereum_DCC_correlation.png` - Bitcoin与Ethereum的动态相关性

## 🔬 模型说明

### GARCH(1,1)模型

$$r_t = \mu + \epsilon_t$$

$$\epsilon_t = \sigma_t z_t, \quad z_t \sim N(0,1)$$

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

其中：
- $\omega > 0$：常数项
- $\alpha \geq 0$：ARCH效应（冲击持续性）
- $\beta \geq 0$：GARCH效应（波动率持续性）
- $\alpha + \beta < 1$：平稳性条件

### DCC-GARCH模型

两阶段估计：
1. **第一阶段**：对每个资产估计边际GARCH模型，得到标准化残差
2. **第二阶段**：基于标准化残差估计动态条件相关系数

$$Q_t = (1 - a - b)\bar{Q} + a(z_{t-1}z_{t-1}') + bQ_{t-1}$$

$$R_t = \text{diag}(Q_t)^{-1/2} Q_t \text{diag}(Q_t)^{-1/2}$$

## 📝 关键发现

1. **模型平稳性**：所有GARCH(1,1)模型均通过平稳性检验（$\alpha + \beta < 1$）

2. **波动率聚集效应**：
   - 传统ETF的$\alpha$值普遍在0.08-0.12之间，表现出明显的波动率聚集
   - Bitcoin的$\alpha$较低（0.05），但$\beta$很高（0.91），波动率持续性强

3. **资产相关性**：
   - **Bitcoin与传统ETF（SPY、IWM）呈弱负相关**，可能具有分散化效益
   - **Bitcoin与Ethereum高度正相关**（ρ ≈ 0.81），表明加密货币市场联动性强

4. **Ethereum特殊性**：Ethereum的GARCH持久性较低（0.51），波动率模式与Bitcoin显著不同

## 📚 数据来源

- **ETF数据**：Wall Street Journal (WSJ)
- **加密货币数据**：CoinMarketCap
- **V-Lab数据**：NYU V-Lab（GARCH、EGARCH、GJR-GARCH模型参考）

## 👥 作者

Financial Data Science HW2 Team

## 📄 许可证

本项目仅用于学术研究目的。
