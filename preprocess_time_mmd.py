import os
import pandas as pd
import numpy as np

BASE_DIR = "/workspace/jinjin/UniCA-master/UniCA-master/data/time-mmd"

DOMAINS = {
    "Climate": "Climate",
    "Energy": "Energy",
    "Environment": "Environment",
    "Public_Health": "Health_US",
    "Security": "Security",
    "SocialGood": "SocialGood",
    "Traffic": "Traffic",
}

for domain, src_name in DOMAINS.items():
    print(f"\nProcessing {domain}...")
    
    # 读取 numerical 数据
    src_dir = os.path.join(BASE_DIR, src_name)
    if not os.path.exists(src_dir):
        print(f"  Skip: {src_dir} not found")
        continue
    
    csvs = [f for f in os.listdir(src_dir) if f.endswith('.csv')]
    if not csvs:
        print(f"  Skip: no CSV in {src_dir}")
        continue
    
    num_file = os.path.join(src_dir, csvs[0])
    df_num = pd.read_csv(num_file)
    print(f"  Numerical: {df_num.shape}")
    
    # 计算 prior_history_avg
    drought_cols = ['D0', 'D1', 'D2', 'D3', 'D4']
    available = [c for c in drought_cols if c in df_num.columns]
    if available:
        df_num['prior_history_avg'] = df_num[available].mean(axis=1)
        print(f"  prior_history_avg from {available}")
    else:
        df_num['prior_history_avg'] = 0.0
    
    # 读取 textual 数据
    text_file = os.path.join(BASE_DIR, "textual", src_name, f"{src_name}_search.csv")
    if os.path.exists(text_file):
        df_text = pd.read_csv(text_file)
        print(f"  Textual: {df_text.shape}")
        
        # 按 start_date 合并
        df_num['start_date'] = pd.to_datetime(df_num['start_date'])
        df_text['start_date'] = pd.to_datetime(df_text['start_date'])
        
        text_col = 'preds' if 'preds' in df_text.columns else 'fact'
        df_merged = df_num.merge(
            df_text[['start_date', text_col]].rename(columns={text_col: 'Final_Search_2'}),
            on='start_date',
            how='left'
        )
        df_merged['Final_Search_2'] = df_merged['Final_Search_2'].fillna("")
    else:
        print(f"  Warning: no textual data")
        df_merged = df_num.copy()
        df_merged['Final_Search_2'] = ""
    
    # 确保 date 列存在
    if 'date' not in df_merged.columns and 'MapDate' in df_merged.columns:
        df_merged.rename(columns={'MapDate': 'date'}, inplace=True)
    if 'date' not in df_merged.columns:
        df_merged['date'] = df_merged['start_date'].astype(str)
    
    # 保存
    out_dir = os.path.join(BASE_DIR, domain)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{domain}.csv")
    df_merged.to_csv(out_file, index=False)
    print(f"  Saved: {out_file}")

print("\nDone!")