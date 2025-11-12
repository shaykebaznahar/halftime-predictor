#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚽ HALF-TIME PREDICTION MODEL BUILDER
Trains 17 Half-Time models from historical data
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

print("\n" + "=" * 80)
print("⚽ HALF-TIME PREDICTION MODEL BUILDER")
print("=" * 80)

# 1. Load data
print("\n📥 Loading data...")
train_df = pd.read_csv('/mnt/user-data/uploads/all_leagues_2005_2019.csv', on_bad_lines='skip')
test_df = pd.read_csv('/mnt/user-data/uploads/all_leagues_2020_2024.csv', on_bad_lines='skip')
print(f"   ✅ Training: {len(train_df)} matches")
print(f"   ✅ Test: {len(test_df)} matches")

# 2. Define features and target
features = ['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'Home_Percentile', 'Away_Percentile']
target = 'HTR'  # Half-Time Result

# 3. Get unique leagues
leagues = sorted(train_df['Div'].unique())
print(f"\n🏆 Leagues found: {len(leagues)}")
print(f"   {', '.join(leagues)}")

# 4. Create output directories
os.makedirs('/home/claude/models_ht', exist_ok=True)
os.makedirs('/home/claude/stats_ht', exist_ok=True)

print("\n" + "=" * 80)
print("🎓 TRAINING MODELS")
print("=" * 80)

# Store results
results = {}

for league in leagues:
    print(f"\n🔄 Processing {league}...")
    
    # Filter data for this league
    train_league = train_df[train_df['Div'] == league].copy()
    test_league = test_df[test_df['Div'] == league].copy()
    
    print(f"   Train matches: {len(train_league)}")
    print(f"   Test matches: {len(test_league)}")
    
    # Remove rows with missing values in features or target
    train_league = train_league.dropna(subset=features + [target])
    test_league = test_league.dropna(subset=features + [target])
    
    print(f"   After cleaning: Train={len(train_league)}, Test={len(test_league)}")
    
    if len(train_league) < 50 or len(test_league) < 10:
        print(f"   ⚠️  Not enough data for {league}, skipping...")
        continue
    
    # Prepare features and target
    X_train = train_league[features]
    y_train = train_league[target]
    X_test = test_league[features]
    y_test = test_league[target]
    
    # Train model
    print(f"   🤖 Training GradientBoosting...")
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        verbose=0
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"   📊 Train Accuracy: {train_score:.3f}")
    print(f"   📊 Test Accuracy: {test_score:.3f}")
    
    # Save model
    model_path = f'/home/claude/models_ht/model_{league}_HT.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"   💾 Model saved: {model_path}")
    
    # Calculate team statistics for current season (from test set, most recent)
    print(f"   📈 Calculating team statistics...")
    
    # Home stats
    home_stats = train_league.groupby('HomeTeam')[features].mean()
    home_stats_path = f'/home/claude/stats_ht/home_stats_{league}_HT_2025.csv'
    home_stats.to_csv(home_stats_path)
    print(f"      Home stats: {len(home_stats)} teams")
    
    # Away stats
    away_stats = train_league.groupby('AwayTeam')[features].mean()
    away_stats_path = f'/home/claude/stats_ht/away_stats_{league}_HT_2025.csv'
    away_stats.to_csv(away_stats_path)
    print(f"      Away stats: {len(away_stats)} teams")
    
    results[league] = {
        'train_acc': train_score,
        'test_acc': test_score,
        'train_matches': len(train_league),
        'test_matches': len(test_league),
        'home_teams': len(home_stats),
        'away_teams': len(away_stats)
    }

# Summary
print("\n" + "=" * 80)
print("✅ SUMMARY")
print("=" * 80)

summary_df = pd.DataFrame(results).T
print(f"\n{summary_df.to_string()}")

print(f"\n✅ Models saved to: /home/claude/models_ht/")
print(f"✅ Stats saved to: /home/claude/stats_ht/")
print(f"✅ Total leagues processed: {len(results)}")

print("\n" + "=" * 80)
print("🎉 DONE!")
print("=" * 80 + "\n")