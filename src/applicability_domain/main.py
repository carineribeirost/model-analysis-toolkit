import argparse
import pandas as pd
import sys
import os
from typing import List

from .calculator import ADCalculator
from .plotting import ADPlotter

def main():
    parser = argparse.ArgumentParser(description="Applicability Domain Analysis Toolkit")
    
    parser.add_argument("--train", required=True, help="Path to the training (internal) CSV file")
    parser.add_argument("--eval", nargs="+", required=True, help="Path(s) to the evaluation (external) CSV file(s)")
    parser.add_argument("--descriptors", nargs="+", required=True, help="List of column names to use as descriptors")
    parser.add_argument("--output-dir", default="ad_results", help="Directory to save results and plots")
    parser.add_argument("--neighbors", type=int, default=10, help="Number of neighbors for KNN (default: 10)")
    
    args = parser.parse_args()

    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"--- Loading Training Data: {args.train} ---")
    try:
        train_df = pd.read_csv(args.train)
        # Check if all descriptors exist
        missing_train = [c for c in args.descriptors if c not in train_df.columns]
        if missing_train:
            print(f"Error: Missing descriptors in training file: {missing_train}")
            sys.exit(1)
            
        train_features = train_df[args.descriptors]
    except Exception as e:
        print(f"Error loading training file: {e}")
        sys.exit(1)

    # Initialize Calculator and Plotter
    calculator = ADCalculator(train_features, n_neighbors=args.neighbors)
    plotter = ADPlotter()

    print(f"KNN Threshold: {calculator.knn_threshold:.4f}")
    print(f"Leverage Threshold: {calculator.leverage_threshold:.4f}")

    # Assess Internal Set
    print("\n--- Assessing Internal Set ---")
    internal_results = calculator.assess(train_features, dataset_name="Internal", is_internal=True)
    
    # Assess External Sets
    all_results_list = [internal_results]
    
    for eval_path in args.eval:
        print(f"--- Assessing External Set: {eval_path} ---")
        try:
            eval_df = pd.read_csv(eval_path)
            # Check descriptors
            missing_eval = [c for c in args.descriptors if c not in eval_df.columns]
            if missing_eval:
                print(f"Warning: Missing descriptors in {eval_path}. Skipping.")
                continue
                
            eval_features = eval_df[args.descriptors]
            ds_name = os.path.basename(eval_path).replace(".csv", "")
            
            eval_results = calculator.assess(eval_features, dataset_name=ds_name, is_internal=False)
            all_results_list.append(eval_results)
            
            # Save individual external result
            eval_results.to_csv(os.path.join(args.output_dir, f"{ds_name}_ad_assessment.csv"))
            
        except Exception as e:
            print(f"Error processing {eval_path}: {e}")

    # Combine all results
    combined_results = pd.concat(all_results_list)
    combined_results.to_csv(os.path.join(args.output_dir, "all_ad_results.csv"))

    # Generate Plots
    print("\n--- Generating Plots ---")
    plotter.plot_distributions(
        combined_results, 
        calculator.knn_threshold, 
        calculator.leverage_threshold,
        save_path=os.path.join(args.output_dir, "distributions.png")
    )
    
    plotter.plot_summary_bar(
        combined_results,
        save_path=os.path.join(args.output_dir, "summary_bars.png")
    )
    
    plotter.plot_williams_ad(
        combined_results,
        calculator.knn_threshold, 
        calculator.leverage_threshold,
        save_path=os.path.join(args.output_dir, "ad_williams_plot.png")
    )

    print(f"\n✓ Analysis complete. Results saved in: {args.output_dir}")
    print(f"  - CSV: all_ad_results.csv")
    print(f"  - Plots: distributions.png, summary_bars.png, ad_williams_plot.png")

if __name__ == "__main__":
    main()
