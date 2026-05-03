import matplotlib.pyplot as plt
import numpy as np
import shap
import os
import scienceplots

class SHAPPlotter:
    """
    Generates high-quality SHAP visualizations.
    """
    def __init__(self, style=['science', 'ieee', 'no-latex']):
        """
        Initialize the plotter with visual styles.
        """
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use('seaborn-v0_8-paper')
        
        # Define custom colors from the notebook
        self.custom_colors = [
            np.array([0, 139, 251]) / 255,  # Blue-ish
            np.array([255, 0, 81]) / 255    # Pink-ish
        ]
        self.cmap = plt.cm.colors.ListedColormap(self.custom_colors)

    def plot_importance(self, explanation: shap.Explanation, max_display: int = 20, save_path: str = "shap_importance.png"):
        """
        Generates a global feature importance bar plot.
        """
        plt.figure(figsize=(10, 8))
        try:
            shap.plots.bar(explanation, max_display=max_display, show=False)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Warning: SHAP bar plot failed ({e}). Attempting manual fallback plot...")
            plt.clf()
            # Manual fallback: Mean absolute SHAP values
            feature_names = explanation.feature_names
            # Handle potential 3D values (unlikely here but for safety)
            values = explanation.values
            if len(values.shape) == 3:
                values = values[:, :, 1]
            
            mean_abs_shap = np.mean(np.abs(values), axis=0)
            idx = np.argsort(mean_abs_shap)[-max_display:]
            
            plt.barh([feature_names[i] for i in idx], mean_abs_shap[idx], color=self.custom_colors[0])
            plt.xlabel("mean(|SHAP value|) (average impact on model output magnitude)")
            plt.title("Feature Importance")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        finally:
            plt.close()

    def plot_beeswarm(self, explanation: shap.Explanation, max_display: int = 31, save_path: str = "shap_beeswarm.png"):
        """
        Generates a standard beeswarm plot.
        """
        plt.figure(figsize=(10, 10))
        try:
            shap.plots.beeswarm(explanation, max_display=max_display, color=self.cmap, show=False)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Warning: SHAP beeswarm plot failed ({e}).")
        finally:
            plt.close()

    def plot_normalized_beeswarm(self, normalized_explanation: shap.Explanation, max_display: int = 31, save_path: str = "shap_beeswarm_normalized.png"):
        """
        Generates a beeswarm plot using normalized SHAP values.
        """
        plt.figure()
        shap.plots.beeswarm(
            normalized_explanation, 
            max_display=max_display, 
            color=self.cmap, 
            group_remaining_features=False, 
            show=False
        )
        plt.gcf().set_size_inches(10, 10)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_force_plots(self, explanation: shap.Explanation, output_dir: str, num_samples: int = 5):
        """
        Saves individual force plots for the first N samples as HTML files.
        """
        force_dir = os.path.join(output_dir, "force_plots")
        if not os.path.exists(force_dir):
            os.makedirs(force_dir)

        for i in range(min(num_samples, len(explanation))):
            plot = shap.plots.force(explanation[i], show=False)
            save_path = os.path.join(force_dir, f"sample_{i}_force_plot.html")
            shap.save_html(save_path, plot)
