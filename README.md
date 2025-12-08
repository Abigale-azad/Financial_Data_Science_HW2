

---

## README: 金融数据科学作业 2

### 1. 项目简介 (Project Overview) 

本项目是为了完成 **金融数据科学作业 2**，主要任务是利用 GARCH 类模型（包括 GARCH(1,1)、GJR-GARCH(1,1) 和 EGARCH(1,1)）以及多变量 GARCH 模型（DCC-GARCH）对不同 ETF 和加密货币的日回报率进行波动率估计和预测 [cite: 22, 23, 30, 32, 34]。

代码文件 `process_wsj_data.py`用于数据预处理，修复 WSJ 历史价格数据中的混合日期格式错误。

### 2. 目录结构说明 (Directory Structure) 

项目根目录位于 `E:\金融数据科学_作业2`。

| 路径 | 描述 |
| :--- | :--- |
| **./** | **E:\金融数据科学\_作业2**：项目根目录，包含主要的 Python 脚本和配置文件。 |
| **data/** | 包含所有原始数据和处理后的数据。 |
| **data/Raw\_data/** | 原始数据的存放目录，按数据来源细分。 |
| **data/Raw\_data/WSJ/** | 包含从华尔街日报（WSJ）获取的 ETF 历史价格数据。 |
| **data/Raw\_data/vlab/** | 包含从 NYU Stern Vlab 获取的 GARCH/EGARCH/GJR-GARCH 模型估计结果文件，用于结果比对和参考。 |
| **data/Raw\_data/coin/** | 包含从 CoinMarketCap 获取的加密货币历史数据。 |
| **data/Processed\_data/** | 存放经过日期清洗和回报率计算等预处理步骤后的最终数据文件。 |
| **process_wsj_data.py** | **数据清洗脚本**：用于处理 `data/Raw_data/WSJ/` 中 CSV 文件的混合日期格式问题。 |

---

### 3. 数据文件介绍 (Data File Descriptions) 

根据原始数据来源，数据文件被分为三类：

#### A. WSJ (华尔街日报) 数据

[cite_start]这些文件是从华尔街日报（WSJ）网页上爬取的 ETF 历史价格数据 [cite: 8, 9, 10, 12, 13, 14]。它们是进行 GARCH 模型估计的基础数据。

| 文件名 | 对应 ETF | 作业对应问题 |
| :--- | :--- | :--- |
| **HistoricalPrices\_S&P 500                                  | S&P 500 SPDR ETF                                | Q1, Q2, Q3 |
| **HistoricalPrices\_Russell 2000 ETF.csv** | [cite_start]Russell 2000 iShares ETF [cite: 10] | Q1, Q2, Q3 |
| [cite_start]**HistoricalPrices\_MSCI Germany ETF.csv** | iShares MSCI Germany [cite: 11] | Q1, Q2 |
| [cite_start]**HistoricalPrices\_MSCI Hong Kong ETF.csv** | iShares MSCI Hong Kong [cite: 13] | Q1, Q2 |
| [cite_start]**HistoricalPrices\_MSCI United Kingdom ETF.csv** | iShares MSCI UK [cite: 14] | Q1, Q2 |

---

#### B. Vlab (NYU Stern Vlab) 数据

文件包含了从 Vlab 获得的 GARCH 族模型的**预估结果**，用于与我们在 Python 中实现的模型结果进行比较和验证。文件名以 **`YYYYMMDD`** 开头（例如 `20251205`），后跟资产名称和模型类型 (`GARCH`, `EGARCH`, `GJR-GARCH`)。

| 文件名示例 | 资产名称 | 模型类型 | 样本范围 |
| :--- | :--- | :--- | :--- |
| `*_iShares安硕罗素2000 ETF_GARCH_*.csv` | Russell 2000 iShares ETF | [cite_start]GARCH(1,1) [cite: 22] | 2000-05-30 至 2025-12-05 |
| `*_MSCI德国指数ETF_EGARCH_*.csv` | iShares MSCI Germany | [cite_start]EGARCH(1,1) [cite: 23] | 1996-04-02 至 2025-12-05 |

---

#### C. Coin (加密货币) 数据

这些文件包含两种主流加密货币的历史价格，用于 Q3 的 GARCH 模型估计 。

| 文件名 | 对应加密货币 | 作业对应问题 |
| :--- | :--- | :--- |
| **Bitcoin\_2024\_12\_8-2025\_12\_8\_historical\_data\_coinmarketcap.csv** | Bitcoin (BTC) | Q3 |
| **Ethereum\_2024\_12\_8-2025\_12\_8\_historical\_data\_coinmarketcap.csv** | Ethereum (ETH) | Q3 |

---

### 4. 代码说明 (Code Description) 

* `process_wsj_data.py`：
    * **作用**：执行数据预处理。
    * **核心功能**：读取 `data/Raw_data/WSJ/` 目录下的所有 CSV 文件，应用自定义的 `clean_mixed_dates` 函数修复其中**混合的日期格式**，并将清洗后的文件保存到 `data/Processed_data/WSJ/` 目录下。

* **主分析脚本** 