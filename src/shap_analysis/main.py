import argparse
import pandas as pd
import os
import sys
import pickle

from .explainer import SHAPExplainer
from .plotting import SHAPPlotter

def main():
    parser = argparse.ArgumentParser(description="SHAP Analysis Toolkit")
    
    parser.add_argument("--models", nargs="+", required=True, help="Path(s) to model pickle file(s)")
    parser.add_argument("--data", required=True, help="Path to the evaluation CSV file")
    parser.add_argument("--target", help="Name of the target column to drop from features (if present)")
    parser.add_argument("--task", default="classification", choices=["classification", "regression"], help="Type of ML task")
    parser.add_argument("--output-dir", default="shap_results", help="Directory to save results and plots")
    parser.add_argument("--max-display", type=int, default=31, help="Max features to display in plots")
    parser.add_argument("--force-samples", type=int, default=5, help="Number of force plots to generate")

    args = parser.parse_args()

    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"--- Loading Data: {args.data} ---")
    try:
        df = pd.read_csv(args.data)
        if args.target and args.target in df.columns:
            X = df.drop(columns=[args.target])
            print(f"Dropped target column: {args.target}")
        else:
            X = df
            print("Using all columns as features.")
            
        # Ensure data is numeric for SHAP
        X = X.select_dtypes(include=['number'])
        print(f"Input shape: {X.shape}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # Initialize SHAP Explainer
    print(f"--- Calculating SHAP Values (Ensemble size: {len(args.models)}) ---")
    try:
        explainer = SHAPExplainer(args.models, task=args.task)
        explanation = explainer.get_ensemble_explanation(X)
        
        # Save raw SHAP values for persistence
        with open(os.path.join(args.output_dir, "ensemble_explanation.pkl"), "wb") as f:
            pickle.dump(explanation, f)
            
    except Exception as e:
        print(f"Error during SHAP calculation: {e}")
        sys.exit(1)

    # Initialize Plotter
    plotter = SHAPPlotter()

    print("--- Generating Plots ---")
    
    # 1. Feature Importance Bar
    plotter.plot_importance(
        explanation, 
        max_display=min(20, args.max_display),
        save_path=os.path.join(args.output_dir, "shap_importance.png")
    )

    # 2. Standard Beeswarm
    plotter.plot_beeswarm(
        explanation,
        max_display=args.max_display,
        save_path=os.path.join(args.output_dir, "shap_beeswarm.png")
    )

    # 3. Normalized Beeswarm
    print("Generating normalized beeswarm...")
    normalized_exp = SHAPExplainer.normalize_explanation(explanation)
    plotter.plot_normalized_beeswarm(
        normalized_exp,
        max_display=args.max_display,
        save_path=os.path.join(args.output_dir, "shap_beeswarm_normalized.png")
    )

    # 4. Force Plots
    print(f"Generating {args.force_samples} local force plots...")
    plotter.save_force_plots(
        explanation,
        output_dir=args.output_dir,
        num_samples=args.force_samples
    )

    print(f"\n✓ SHAP Analysis complete. Results saved in: {args.output_dir}")
    print(f"  - Importance Plot: shap_importance.png")
    print(f"  - Beeswarm Plots: shap_beeswarm.png, shap_beeswarm_normalized.png")
    print(f"  - Force Plots: force_plots/ directory")
    print(f"  - Data: ensemble_explanation.pkl")

if __name__ == "__main__":
    main()
