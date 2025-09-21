import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Dict, Callable, Type
import warnings

# Assuming other scripts are in the same directory or a configured src folder
from src.sampling_methods import SamplingMethod, HierarchicalBayesianSampling, PureBayesianSampling
from src.objectives import sharpe_performance_function, calmar_performance_function
from src.testing_improved import TestEnvironment, AssetDataGenerator

warnings.filterwarnings('ignore', category=RuntimeWarning)


class ComparativeExperimentRunner:
    """
    Runs a robust, scalable experiment to compare two sampling methods by
    focusing on the relative percentage gain within each randomized trial.
    """
    
    def __init__(self,
                 method_tree: Type[SamplingMethod],
                 method_input: Type[SamplingMethod],
                 performance_function: Callable,
                 scaling_variable: str,
                 scaling_values: List[int],
                 num_trials: int = 5,
                 fixed_n_assets: int = 50,
                 fixed_time_limit: int = 30):
        """Initializes the experiment runner."""
        self.method_tree_class = method_tree
        self.method_input_class = method_input
        self.performance_function = performance_function
        
        if scaling_variable not in ['n_assets', 'sampling_time']:
            raise ValueError("scaling_variable must be 'n_assets' or 'sampling_time'")
            
        self.scaling_variable = scaling_variable
        self.scaling_values = sorted(scaling_values)
        self.num_trials = num_trials
        
        self.fixed_n_assets = fixed_n_assets
        self.fixed_time_limit = fixed_time_limit
        
        self.results = []
        self._asset_pool_data = None
        self._prepare_asset_pool()
        
    def _prepare_asset_pool(self, pool_size: int = 500, history: str = "10y"):
        """Downloads a large pool of asset data once to be sampled from later."""
        print(f"Downloading a large asset pool (~{pool_size} assets, {history} history) for robust trials...")
        # Note: This now calls the updated GetSP500Assets in testing_improved.py
        self._asset_pool_data = AssetDataGenerator.get_sp500_assets(pool_size, period=history)
        print("Asset pool is ready.")

    def _get_robust_data(self, n_assets: int, period_length_days: int = 504) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
        """
        Generates a randomized dataset for a single trial by sampling
        random tickers and a random time slice from the main pool.
        """
        pool_returns, pool_tickers, pool_prices = self._asset_pool_data
        
        # Select random tickers
        selected_tickers = np.random.choice(pool_tickers, n_assets, replace=False).tolist()
        
        # Select a random time period
        if len(pool_returns) <= period_length_days:
            start_idx = 0
        else:
            start_idx = np.random.randint(0, len(pool_returns) - period_length_days)
        end_idx = start_idx + period_length_days
        
        # Slice data for the trial
        trial_returns = pool_returns.iloc[start_idx:end_idx][selected_tickers].copy()
        trial_prices = pool_prices.iloc[start_idx:end_idx][selected_tickers].copy()
        
        # Ensure data is clean for this specific slice
        trial_returns.dropna(axis=1, how='all', inplace=True)
        trial_prices = trial_prices[trial_returns.columns] # Align prices with valid returns columns
        
        return trial_returns, list(trial_returns.columns), trial_prices

    def run(self):
        """Executes the full comparative experiment."""
        print(f"\n{'='*80}")
        print(f"Starting Experiment: Comparing {self.method_tree_class.__name__} vs. {self.method_input_class.__name__}")
        print(f"Scaling by: {self.scaling_variable.replace('_', ' ').title()}")
        print(f"Objective: {self.performance_function.__name__}")
        print(f"Scaling values: {self.scaling_values}")
        print(f"Trials per value: {self.num_trials}")
        print(f"{'='*80}")
        
        for val in self.scaling_values:
            for i in range(self.num_trials):
                print(f"\n--- Running trial {i+1}/{self.num_trials} for {self.scaling_variable} = {val} ---")
                
                if self.scaling_variable == 'n_assets':
                    n_assets, time_limit = val, self.fixed_time_limit
                else: # 'sampling_time'
                    n_assets, time_limit = self.fixed_n_assets, val

                try:
                    returns, assets, prices = self._get_robust_data(n_assets)
                    actual_n_assets = len(assets)
                    if actual_n_assets < n_assets * 0.9:
                        print(f"Warning: Could only fetch {actual_n_assets}/{n_assets} assets for trial. Skipping.")
                        continue
                except Exception as e:
                    print(f"Error generating data for trial: {e}. Skipping.")
                    continue

                env = TestEnvironment(
                    n_assets=actual_n_assets, performance_function=self.performance_function,
                    use_real_data=True, returns_data=returns, asset_names=assets, price_data=prices
                )
                
                # Run Tree Method
                res_tree = env.run_experiment(self.method_tree_class(), time_limit)
                score_tree = res_tree['final_best_score']
                print(f"  -> {res_tree['method_name']} | Score: {score_tree:.4f}")
                
                # Run Input Space Method
                res_input = env.run_experiment(self.method_input_class(), time_limit)
                score_input = res_input['final_best_score']
                print(f"  -> {res_input['method_name']} | Score: {score_input:.4f}")

                # Calculate percent gain for this trial
                if score_input is not None and score_tree is not None and abs(score_input) > 1e-9:
                    percent_gain = 100 * (score_tree - score_input) / abs(score_input)
                    print(f"  => Trial Percent Gain for Tree Method: {percent_gain:+.2f}%")
                    self.results.append({
                        'scaling_value': val,
                        'trial': i,
                        'percent_gain': percent_gain,
                    })
        
        self.results_df = pd.DataFrame(self.results)
        self._analyze_and_plot()

    def _analyze_and_plot(self):
        """Analyzes the results and generates the percent gain plot."""
        if not hasattr(self, 'results_df') or self.results_df.empty:
            print("\nNo results to analyze. This could happen if trials failed to complete.")
            return

        summary = self.results_df.groupby('scaling_value')['percent_gain'].agg(['mean', 'std']).reset_index()
        self.plot_percent_difference(summary)

    def plot_percent_difference(self, summary_df: pd.DataFrame):
        """Plots the average percent gain with a standard deviation error band."""
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 7))

        method_tree_name = self.method_tree_class().get_name()

        ax.plot(summary_df['scaling_value'], summary_df['mean'], color='green', marker='D', markersize=8, 
                label=f'Avg. Gain of {method_tree_name}')
        
        ax.fill_between(summary_df['scaling_value'], 
                        summary_df['mean'] - summary_df['std'], 
                        summary_df['mean'] + summary_df['std'], 
                        color='green', alpha=0.15, label='Gain (1 Std Dev)')
        
        ax.axhline(0, color='black', linestyle='--', linewidth=1.5, label='No Improvement')
        
        ax.set_xlabel(f"Scaling Variable: {self.scaling_variable.replace('_', ' ').title()}", fontsize=12)
        ax.set_ylabel(f"Average Percent Gain (%)", fontsize=12)
        ax.set_title(f"Relative Performance Gain of Tree Topology vs. Input Space", fontsize=14, weight='bold')
        ax.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # --- EXPERIMENT 1: SCALING BY NUMBER OF ASSETS ---
    print("\n\n*** CONFIGURING EXPERIMENT 1: SCALING BY NUMBER OF ASSETS ***")
    exp1 = ComparativeExperimentRunner(
        method_tree=HierarchicalBayesianSampling,
        method_input=PureBayesianSampling,
        performance_function=sharpe_performance_function,
        scaling_variable='n_assets',
        scaling_values=[10, 25, 50, 100, 250, 500], # X-axis values
        num_trials=5,                          # Use 5-10 for robust results
        fixed_time_limit=60                    # Fixed time for each run
    )
    exp1.run()
    
    # --- EXPERIMENT 2: SCALING BY SAMPLING TIME ---
    print("\n\n*** CONFIGURING EXPERIMENT 2: SCALING BY SAMPLING TIME ***")
    exp2 = ComparativeExperimentRunner(
        method_tree=HierarchicalBayesianSampling,
        method_input=PureBayesianSampling,
        performance_function=sharpe_performance_function,
        scaling_variable='sampling_time',
        scaling_values=[20, 40, 60, 90, 120],  # X-axis values (seconds)
        num_trials=5,                           # Use 5-10 for robust results
        fixed_n_assets=50                       # Fixed number of assets
    )
    exp2.run()