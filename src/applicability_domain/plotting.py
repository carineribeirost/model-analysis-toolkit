import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scienceplots

class ADPlotter:
    """
    Generates high-quality visualizations for Applicability Domain analysis.
    """
    def __init__(self, style=['science', 'ieee', 'no-latex']):
        """
        Initialize the plotter with a specific style.
        """
        try:
            plt.style.use(style)
        except Exception:
            # Fallback to a basic style if scienceplots is not configured correctly
            plt.style.use('seaborn-v0_8-paper')
        
        # Set default font to ensure consistency
        plt.rcParams['font.family'] = 'serif'

    def plot_distributions(self, all_results: pd.DataFrame, knn_threshold: float, leverage_threshold: float, save_path: str = "distributions_plot.png"):
        """
        Plots histograms for KNN Distance and Leverage distributions (Cell 7 equivalent).
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        datasets = all_results['Dataset'].unique()
        colors = sns.color_palette("Set1", len(datasets))

        for i, dataset in enumerate(datasets):
            mask = all_results['Dataset'] == dataset
            
            # KNN Distribution
            axes[0].hist(all_results.loc[mask, 'KNN_Distance'], bins=30, alpha=0.6, 
                         color=colors[i], label=dataset, edgecolor='black')
            
            # Leverage Distribution
            axes[1].hist(all_results.loc[mask, 'Leverage'], bins=30, alpha=0.6, 
                         color=colors[i], label=dataset, edgecolor='black')

        # Formatting KNN Plot
        axes[0].axvline(x=knn_threshold, color='red', linestyle='--', linewidth=1.5, label='Threshold')
        axes[0].set_xlabel('Average KNN Distance', fontweight='bold')
        axes[0].set_ylabel('Frequency', fontweight='bold')
        axes[0].set_title('Distribution of KNN Distances')
        axes[0].legend()

        # Formatting Leverage Plot
        axes[1].axvline(x=leverage_threshold, color='red', linestyle='--', linewidth=1.5, label='Threshold')
        axes[1].set_xlabel('Leverage', fontweight='bold')
        axes[1].set_ylabel('Frequency', fontweight='bold')
        axes[1].set_title('Distribution of Leverage Values')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_summary_bar(self, all_results: pd.DataFrame, save_path: str = "summary_bar_chart.png"):
        """
        Plots a bar chart summary of molecules In vs Out of domain (Cell 8 equivalent).
        """
        summary_data = []
        datasets = all_results['Dataset'].unique()
        
        for ds in datasets:
            mask = all_results['Dataset'] == ds
            in_ad = all_results.loc[mask, 'In_Applicability_Domain'].sum()
            out_ad = mask.sum() - in_ad
            summary_data.append({'Dataset': ds, 'Status': 'In AD', 'Count': in_ad})
            summary_data.append({'Dataset': ds, 'Status': 'Out AD', 'Count': out_ad})
        
        df_summary = pd.DataFrame(summary_data)
        
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=df_summary, x='Dataset', y='Count', hue='Status', palette=['#2ecc71', '#e74c3c'])
        
        # Add values on top of bars
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f'{int(p.get_height())}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='center', 
                            xytext=(0, 9), 
                            textcoords='offset points',
                            fontweight='bold')

        plt.title('Applicability Domain Summary', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Molecules', fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_williams_ad(self, all_results: pd.DataFrame, knn_threshold: float, leverage_threshold: float, save_path: str = "applicability_domain_plot.png"):
        """
        Plots the Williams-style AD plot (Leverage vs KNN Distance).
        """
        fig, ax = plt.subplots(figsize=(7, 5))

        # Define internal vs external for specific coloring
        internal_mask = all_results['Dataset'].str.lower() == 'internal'
        external_mask = ~internal_mask

        # Plot Internal
        if internal_mask.any():
            ax.scatter(all_results.loc[internal_mask, 'KNN_Distance'], 
                       all_results.loc[internal_mask, 'Leverage'],
                       label='Internal Dataset', color='#1f77b4', s=15, alpha=0.7)

        # Plot External
        if external_mask.any():
            ax.scatter(all_results.loc[external_mask, 'KNN_Distance'], 
                       all_results.loc[external_mask, 'Leverage'],
                       label='External Dataset', color='#00CC00', s=15, alpha=0.7)

        # Threshold lines (Williams plot)
        ax.axvline(x=knn_threshold, color='red', linestyle='-.', linewidth=1.5, label='KNN Threshold')
        ax.axhline(y=leverage_threshold, color='red', linestyle='-.', linewidth=1.5, label='Leverage Threshold')

        # Annotations
        max_knn = all_results['KNN_Distance'].max()
        max_lev = all_results['Leverage'].max()
        
        text_str = f"KNN Threshold = {knn_threshold:.4f}\nLeverage Threshold = {leverage_threshold:.4f}"
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black')
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)

        ax.set_xlabel('Average KNN Distance', fontweight='bold')
        ax.set_ylabel('Leverage', fontweight='bold')
        ax.set_title('Applicability Domain Analysis', fontweight='bold')
        
        # Legend outside
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Consistent limits - ensure they cover both data and thresholds
        ax.set_xlim(left=0, right=max(max_knn, knn_threshold) * 1.1)
        ax.set_ylim(bottom=0, top=max(max_lev, leverage_threshold) * 1.2)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
