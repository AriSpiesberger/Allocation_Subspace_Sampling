# comparative_experiment.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Dict, Callable, Type

# Assuming other scripts are in the same directory or a configured src folder
from src.sampling_methods import SamplingMethod, HierarchicalBayesianSampling, PureBayesianSampling
from src.objectives import sharpe_performance_function, calmar_performance_function
from src.testing_improved import TestEnvironment, AssetDataGenerator

class ComparativeExperimentRunner:
    """
    Runs a comprehensive, scalable experiment to compare two sampling methods.

    This class is designed to test a "tree topology" method against an 
    "input space" method by scaling a given parameter (e.g., n_assets or
    sampling_time) and running multiple randomized trials to ensure
    statistical robustness.
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
        """
        Initializes the experiment runner.

        Args:
            method_tree: The class of the tree-based sampling method.
            method_input: The class of the input space-based sampling method.
            performance_function: The objective function to optimize.
            scaling_variable: The parameter to scale ('n_assets' or 'sampling_time').
            scaling_values: A list of values for the scaling parameter.
            num_trials: Number of random trials to run for each scaling value.
            fixed_n_assets: The number of assets to use when scaling time.
            fixed_time_limit: The time limit to use when scaling assets.
        """
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
        
    def _prepare_asset_pool(self, pool_size: int = 200, history: str = "10y"):
        """Downloads a large pool of asset data once to be sampled from later."""
        print(f"Downloading a large asset pool ({pool_size} assets, {history} history) for robust trials...")
        self._asset_pool_data = AssetDataGenerator.get_sp500_assets(pool_size, period=history)
        print("Asset pool is ready.")

    def _get_robust_data(self, n_assets: int, period_length_days: int = 504) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
        """
        Generates a randomized dataset for a single trial.
        It samples random tickers and a random time slice from the main pool.
        """
        pool_returns, pool_tickers, pool_prices = self._asset_pool_data
        
        # 1. Select random tickers
        selected_tickers = np.random.choice(pool_tickers, n_assets, replace=False).tolist()
        
        # 2. Select a random time period
        if len(pool_returns) <= period_length_days:
            start_idx = 0
        else:
            start_idx = np.random.randint(0, len(pool_returns) - period_length_days)
        
        end_idx = start_idx + period_length_days
        
        # Slice data for the trial
        trial_returns = pool_returns.iloc[start_idx:end_idx][selected_tickers]
        trial_prices = pool_prices.iloc[start_idx:end_idx][selected_tickers]
        
        return trial_returns.dropna(), list(trial_returns.columns), trial_prices.dropna()

    def run(self):
        """Executes the full comparative experiment."""
        print(f"\n{'='*80}")
        print(f"Starting Experiment: Comparing {self.method_tree_class.__name__} vs. {self.method_input_class.__name__}")
        print(f"Scaling by: {self.scaling_variable}")
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

                # Get a fresh, randomized dataset for this trial
                try:
                    returns, assets, prices = self._get_robust_data(n_assets)
                    if len(assets) != n_assets:
                        print(f"Warning: Could only fetch {len(assets)}/{n_assets} assets for this trial. Skipping.")
                        continue
                except Exception as e:
                    print(f"Error generating data for trial: {e}. Skipping.")
                    continue

                # Initialize methods and environment
                method_tree = self.method_tree_class()
                method_input = self.method_input_class()
                
                env = TestEnvironment(
                    n_assets=n_assets,
                    performance_function=self.performance_function,
                    use_real_data=True,
                    returns_data=returns,
                    asset_names=assets,
                    price_data=prices
                )
                
                # Run and store results for both methods
                for method in [method_tree, method_input]:
                    start_time = time.time()
                    res = env.run_experiment(method, time_limit_seconds=time_limit)
                    run_duration = time.time() - start_time
                    
                    self.results.append({
                        'scaling_variable': self.scaling_variable,
                        'scaling_value': val,
                        'trial': i,
                        'method_name': res['method_name'],
                        'score': res['final_best_score'],
                        'iterations': res['total_iterations'],
                        'duration': run_duration
                    })
                    print(f"  -> {res['method_name']} finished. Score: {res['final_best_score']:.4f}")
        
        self.results_df = pd.DataFrame(self.results)
        self._analyze_and_plot()

    def _analyze_and_plot(self):
        """Analyzes the results and generates the required plots."""
        if not hasattr(self, 'results_df') or self.results_df.empty:
            print("No results to analyze.")
            return

        summary = self.results_df.groupby(['scaling_value', 'method_name'])['score'].agg(['mean', 'std']).reset_index()
        
        self.plot_performance(summary)
        self.plot_percent_difference(summary)

    def plot_performance(self, summary_df: pd.DataFrame):
        """Plots the performance comparison with error bands."""
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 7))

        method_tree_name = self.method_tree_class().get_name()
        method_input_name = self.method_input_class().get_name()
        
        for name, color, marker in [(method_tree_name, 'blue', 'o'), (method_input_name, 'red', 's')]:
            data = summary_df[summary_df['method_name'] == name]
            ax.plot(data['scaling_value'], data['mean'], label=name, color=color, marker=marker, markersize=8)
            ax.fill_between(data['scaling_value'], 
                            data['mean'] - data['std'], 
                            data['mean'] + data['std'], 
                            color=color, alpha=0.15, label=f'{name} (1 Std Dev)')

        ax.set_xlabel(f"Scaling Variable: {self.scaling_variable.replace('_', ' ').title()}", fontsize=12)
        ax.set_ylabel(f"Performance Score ({self.performance_function.__name__})", fontsize=12)
        ax.set_title(f"Performance Comparison: {method_tree_name} vs. {method_input_name}", fontsize=14, weight='bold')
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_percent_difference(self, summary_df: pd.DataFrame):
        """Plots the percent gain of the tree-based method over the input-based method."""
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 7))

        method_tree_name = self.method_tree_class().get_name()
        method_input_name = self.method_input_class().get_name()

        tree_data = summary_df[summary_df['method_name'] == method_tree_name].set_index('scaling_value')
        input_data = summary_df[summary_df['method_name'] == method_input_name].set_index('scaling_value')
        
        # Align dataframes for calculation
        aligned_data = tree_data.join(input_data, lsuffix='_tree', rsuffix='_input')
        
        # Calculate percent difference
        percent_gain = 100 * (aligned_data['mean_tree'] - aligned_data['mean_input']) / abs(aligned_data['mean_input'])
        
        ax.plot(percent_gain.index, percent_gain.values, color='green', marker='D', markersize=8, label='Percent Gain')
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        
        ax.set_xlabel(f"Scaling Variable: {self.scaling_variable.replace('_', ' ').title()}", fontsize=12)
        ax.set_ylabel(f"Percent Gain of {method_tree_name} (%)", fontsize=12)
        ax.set_title(f"Relative Performance Gain of Tree Topology", fontsize=14, weight='bold')
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
        scaling_values=[10, 25, 50, 75, 100], # X-axis values
        num_trials=3,                          # Lower for a quick demo, use 5-10 for real results
        fixed_time_limit=45                    # Fixed time for each run
    )
    exp1.run()
    
    # --- EXPERIMENT 2: SCALING BY SAMPLING TIME ---
    print("\n\n*** CONFIGURING EXPERIMENT 2: SCALING BY SAMPLING TIME ***")
    exp2 = ComparativeExperimentRunner(
        method_tree=HierarchicalBayesianSampling,
        method_input=PureBayesianSampling,
        performance_function=sharpe_performance_function,
        scaling_variable='sampling_time',
        scaling_values=[15, 30, 60, 90, 120],  # X-axis values (seconds)
        num_trials=3,                           # Lower for a quick demo
        fixed_n_assets=40                       # Fixed number of assets
    )
    exp2.run()