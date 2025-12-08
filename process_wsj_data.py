import pandas as pd
from datetime import datetime
from pathlib import Path


# ==============================================================================
# 1. 日期清洗函数
# ==============================================================================

def clean_mixed_dates(date_str):
    """
    清洗混合格式的日期字符串，修复 '20YY/MM/DD' 到 'MM/DD/YY' 的污染问题，
    并扩充两位年份到四位年份（如 99 -> 1999，25 -> 2025）。
    """
    # 检查非字符串或空值
    if pd.isna(date_str) or not isinstance(date_str, str):
        return pd.NaT

    try:
        parts = date_str.split('/')
        # 确保是 M/D/Y 格式
        if len(parts) != 3:
            return pd.NaT

        p1, p2, p3 = parts  # p1: 月份?, p2: 日期, p3: 年份

        # 修复月份 (处理 "2012" -> "12" 的情况)
        month_str = p1
        if len(p1) == 4 and p1.startswith('20'):
            # 如果第一部分是四位数且以 '20' 开头，认为前两位 '20' 是多余的
            month_str = p1[-2:]

        month = int(month_str)
        day = int(p2)

        # 扩充年份 (YY -> YYYY)
        year_short = int(p3)
        # 世纪划分阈值设定为 50：> 50 视为 19XX，<= 50 视为 20XX
        if year_short > 50:
            year = 1900 + year_short
        else:
            year = 2000 + year_short

        # 返回标准的 datetime 对象
        return datetime(year, month, day)

    except Exception:
        # 任何解析错误都返回 Not a Time
        return pd.NaT


# ==============================================================================
# 2. 文件处理主逻辑
# ==============================================================================

def process_wsj_files():
    # 定义基础路径
    # E:\金融数据科学_作业2
    base_dir = Path(r"E:\金融数据科学_作业2")

    # 定义输入和输出路径
    input_dir = base_dir / "data" / "Raw_data" / "WSJ"
    output_dir = base_dir / "data" / "Processed_data" / "WSJ"

    # 确保输出目录存在，如果不存在则创建
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- 开始处理 WSJ 数据 ---")
    print(f"输入路径: {input_dir}")
    print(f"输出路径: {output_dir}")

    processed_count = 0

    # 遍历输入目录下的所有 CSV 文件
    for file_path in input_dir.glob("*.csv"):
        file_name = file_path.name
        print(f"\n正在处理文件: {file_name}")

        try:
            # 1. 读取数据
            # 假设 WSJ 文件默认的日期列名为 'Date'
            df = pd.read_csv(file_path)

            # 2. 检查是否有 'Date' 列
            if 'Date' not in df.columns:
                print(f"警告：文件 {file_name} 中未找到 'Date' 列，跳过日期清洗。")
                continue

            # 3. 应用日期清洗函数
            # 使用 .apply() 方法进行逐行清洗
            df['Date'] = df['Date'].apply(clean_mixed_dates)

            # 4. 检查是否有解析失败的日期，并进行报告
            failed_count = df['Date'].isna().sum() - df['Date'].isnull().sum()
            if failed_count > 0:
                print(f"警告：有 {failed_count} 条日期数据清洗失败并被标记为 NaT。")

            # 5. 保存清洗后的数据
            output_file_path = output_dir / file_name
            df.to_csv(output_file_path, index=False)

            print(f"成功保存到: {output_file_path.name}")
            processed_count += 1

        except Exception as e:
            print(f"处理文件 {file_name} 时发生致命错误: {e}")

    print(f"\n--- WSJ 数据处理完成。共处理 {processed_count} 个文件 ---")


# 运行主函数
if __name__ == "__main__":
    process_wsj_files()