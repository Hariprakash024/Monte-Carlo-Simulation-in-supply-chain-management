#!/usr/bin/env python3
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import argparse, warnings
warnings.filterwarnings('ignore')

class SupplyChainAnalytics:
    def __init__(self, results_csv):
        try:
            self.df = pd.read_csv(results_csv)
            print(f"Loaded {len(self.df)} rows from {results_csv}")
        except FileNotFoundError:
            print(f"Results file not found: {results_csv}")
            self.df = None

    def summary(self):
        if self.df is None: return
        print("\n=== SUMMARY ===")
        if 'total_cost' in self.df:
            print("Mean cost:", self.df['total_cost'].mean())
            print("Std cost:", self.df['total_cost'].std())
            print("95% quantile:", self.df['total_cost'].quantile(0.95))
        if 'service_level' in self.df:
            print("Mean service:", self.df['service_level'].mean()*100, "%")

    def plots(self):
        if self.df is None: return
        if 'total_cost' in self.df:
            plt.figure(figsize=(6,4)); self.df['total_cost'].hist(bins=40); plt.title('Total Cost'); plt.savefig('dist_cost.png', dpi=200); plt.close()
        if 'service_level' in self.df:
            plt.figure(figsize=(6,4)); (self.df['service_level']*100).hist(bins=40); plt.title('Service Level %'); plt.savefig('dist_service.png', dpi=200); plt.close()

    def ml_opt(self, params_csv='simulation_parameters.csv'):
        if self.df is None: return
        try:
            params = pd.read_csv(params_csv)
        except FileNotFoundError:
            print(f"No params file {params_csv}; skipping ML.")
            return
        common = pd.concat([params.reset_index(drop=True), self.df[['total_cost','service_level']].reset_index(drop=True)], axis=1)
        feats = [c for c in params.columns if c not in ['scenario_id']]
        X = common[feats]; y = common['total_cost']
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(Xtr,ytr); pred = rf.predict(Xte); r2=r2_score(yte,pred)
        print(f"RF cost prediction R^2: {r2:.3f}")
        imp = pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=False)
        imp.head(10).to_csv('feature_importance_top10.csv')
        plt.figure(figsize=(6,4)); imp.head(10).plot(kind='barh'); plt.title('Top 10 features'); plt.tight_layout(); plt.savefig('feat_importance.png', dpi=200); plt.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--results', default='simulation_results.csv')
    ap.add_argument('--params', default='simulation_parameters.csv')
    ap.add_argument('--all', action='store_true')
    args=ap.parse_args()
    eng=SupplyChainAnalytics(args.results)
    eng.summary(); eng.plots(); eng.ml_opt(args.params) if args.all else None
