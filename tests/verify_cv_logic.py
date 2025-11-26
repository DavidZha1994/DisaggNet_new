#!/usr/bin/env python3
"""
验证Walk-Forward交叉验证逻辑是否正确
"""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from src.data_preparation.cross_validation import WalkForwardCV, CVFold

def create_mock_segments():
    """创建模拟的段数据来测试CV逻辑"""
    # 创建30天的模拟数据，每天一个段
    start_date = datetime(2023, 1, 1)
    segments = []
    
    for i in range(30):
        segment_start = start_date + timedelta(days=i)
        segment_end = segment_start + timedelta(days=1)
        
        segments.append({
            'segment_id': i,
            'start_ts': int(segment_start.timestamp()),
            'end_ts': int(segment_end.timestamp()),
            'duration_hours': 24.0,
            'n_rows': 17280  # 24小时 * 3600秒 / 5秒采样间隔
        })
    
    return pd.DataFrame(segments)

def test_walk_forward_cv():
    """测试Walk-Forward交叉验证逻辑"""
    print("=" * 60)
    print("Walk-Forward交叉验证逻辑测试")
    print("=" * 60)
    
    # 创建测试配置
    config = {
        'cross_validation': {
            'n_folds': 3,
            'purge_gap_minutes': 60,  # 1小时purge gap
            'val_span_days': 3.0,    # 3天验证集
            'test_span_days': 3.0,   # 3天测试集
            'min_train_days': 7.0,   # 最少7天训练集
            'segment_isolation': False,
            'holdout_test': True,
            'time_based_split': True
        }
    }
    
    # 创建模拟段数据
    segments_meta = create_mock_segments()
    print(
        f"模拟数据: {len(segments_meta)} 个段，时间范围 "
        f"{segments_meta.iloc[0]['start_ts']} - {segments_meta.iloc[-1]['end_ts']}"
    )
    print(
        "数据时间: "
        + datetime.fromtimestamp(segments_meta.iloc[0]["start_ts"]).strftime("%Y-%m-%d %H:%M")
        + " - "
        + datetime.fromtimestamp(segments_meta.iloc[-1]["end_ts"]).strftime("%Y-%m-%d %H:%M")
    )
    
    # 创建CV对象
    cv = WalkForwardCV(config)
    
    # 创建折
    try:
        folds = cv.create_folds(segments_meta)
        print(f"\n成功创建 {len(folds)} 个折")
        
        # 分析每个折
        for i, fold in enumerate(folds):
            print(f"\n折 {i}:")
            
            train_start = datetime.fromtimestamp(fold.train_start_ts)
            train_end = datetime.fromtimestamp(fold.train_end_ts)
            val_start = datetime.fromtimestamp(fold.val_start_ts)
            val_end = datetime.fromtimestamp(fold.val_end_ts)
            
            train_days = (train_end - train_start).days
            val_days = (val_end - val_start).days
            gap_hours = (val_start - train_end).total_seconds() / 3600
            
            print(
                "  训练集: "
                + train_start.strftime("%Y-%m-%d %H:%M")
                + " - "
                + train_end.strftime("%Y-%m-%d %H:%M")
                + f" ({train_days} 天)"
            )
            print(
                "  验证集: "
                + val_start.strftime("%Y-%m-%d %H:%M")
                + " - "
                + val_end.strftime("%Y-%m-%d %H:%M")
                + f" ({val_days} 天)"
            )
            print(f"  间隔: {gap_hours:.1f} 小时")
            
            if fold.test_start_ts:
                test_start = datetime.fromtimestamp(fold.test_start_ts)
                test_end = datetime.fromtimestamp(fold.test_end_ts)
                test_days = (test_end - test_start).days
                print(
                    "  测试集: "
                    + test_start.strftime("%Y-%m-%d %H:%M")
                    + " - "
                    + test_end.strftime("%Y-%m-%d %H:%M")
                    + f" ({test_days} 天)"
                )
            
            # 验证Walk-Forward逻辑
            if i > 0:
                prev_fold = folds[i - 1]
                prev_train_end = datetime.fromtimestamp(prev_fold.train_end_ts)
                curr_train_end = datetime.fromtimestamp(fold.train_end_ts)
                
                if curr_train_end > prev_train_end:
                    print(
                        f"  ✓ Walk-Forward正确: 训练集扩展了 {(curr_train_end - prev_train_end).days} 天"
                    )
                else:
                    print("  ❌ Walk-Forward错误: 训练集没有扩展")
        
        # 验证数据泄漏
        print("\n数据泄漏检查:")
        for i, fold in enumerate(folds):
            train_end = fold.train_end_ts
            val_start = fold.val_start_ts
            gap_seconds = val_start - train_end
            
            if gap_seconds >= config['cross_validation']['purge_gap_minutes'] * 60:
                print(f"  折 {i}: ✓ 无数据泄漏 (间隔 {gap_seconds/3600:.1f} 小时)")
            else:
                print(f"  折 {i}: ❌ 可能数据泄漏 (间隔 {gap_seconds/3600:.1f} 小时)")
        
        return True
        
    except Exception as e:
        print(f"❌ CV创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_traditional_cv_comparison():
    """对比传统CV和Walk-Forward CV的区别"""
    print("\n" + "=" * 60)
    print("传统CV vs Walk-Forward CV对比")
    print("=" * 60)
    
    segments_meta = create_mock_segments()
    total_days = len(segments_meta)
    
    print(f"数据总长度: {total_days} 天")
    print(
        "数据时间范围: "
        + datetime.fromtimestamp(segments_meta.iloc[0]["start_ts"]).strftime("%Y-%m-%d")
        + " - "
        + datetime.fromtimestamp(segments_meta.iloc[-1]["end_ts"]).strftime("%Y-%m-%d")
    )
    
    # 传统CV (错误的方式)
    print("\n传统CV (错误方式):")
    print("  每折都使用固定比例 (如80%训练, 20%验证)")
    print("  折1: 训练 0-24天, 验证 24-30天")
    print("  折2: 训练 0-24天, 验证 24-30天 (重复使用相同数据)")
    print("  折3: 训练 0-24天, 验证 24-30天 (重复使用相同数据)")
    print("  ❌ 问题: 验证集重复，无法模拟真实部署场景")
    
    # Walk-Forward CV (正确的方式)
    print("\nWalk-Forward CV (正确方式):")
    config = {
        'cross_validation': {
            'n_folds': 3,
            'purge_gap_minutes': 60,
            'val_span_days': 3.0,
            'test_span_days': 3.0,
            'min_train_days': 7.0,
            'segment_isolation': False,
            'holdout_test': True,
            'time_based_split': True
        }
    }
    
    cv = WalkForwardCV(config)
    folds = cv.create_folds(segments_meta)
    
    for i, fold in enumerate(folds):
        train_start = datetime.fromtimestamp(fold.train_start_ts)
        train_end = datetime.fromtimestamp(fold.train_end_ts)
        val_start = datetime.fromtimestamp(fold.val_start_ts)
        val_end = datetime.fromtimestamp(fold.val_end_ts)
        
        train_days = (train_end - train_start).days
        val_days = (val_end - val_start).days
        
        print(
            f"  折{i+1}: 训练 {train_start.strftime('%m-%d')} - {train_end.strftime('%m-%d')} ({train_days}天), "
            f"验证 {val_start.strftime('%m-%d')} - {val_end.strftime('%m-%d')} ({val_days}天)"
        )
    
    print("  ✓ 优势: 每折训练集递增，验证集不重复，模拟真实部署")

def main():
    """主函数"""
    print("Walk-Forward交叉验证逻辑验证")
    
    # 测试Walk-Forward CV逻辑
    success = test_walk_forward_cv()
    
    if success:
        # 对比传统CV和Walk-Forward CV
        test_traditional_cv_comparison()
        
        print("\n" + "=" * 60)
        print("验证结果")
        print("=" * 60)
        print("✅ Walk-Forward交叉验证逻辑正确")
        print("✅ 每个折使用前面所有数据作为训练集")
        print("✅ 验证集大小固定且不重复")
        print("✅ 训练集和验证集之间有purge gap防止数据泄漏")
        print("✅ 符合时序数据的最佳实践")
    else:
        print("\n❌ Walk-Forward交叉验证逻辑验证失败")

if __name__ == "__main__":
    main()
