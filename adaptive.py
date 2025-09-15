# -*- coding: utf-8 -*-
"""
Focused Portfolio Optimization Experiments
REFACTORED to use a single, consistent dataset for all experiments.
This ensures fair comparison across all objectives and methods by preventing
data readjustment between runs.
@author: AriSpiesberger
"""
from src.sampling_methods import *
from src.objectives import *
from src.testing_improved import TestEnvironment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def run_focused_experiments():
    """
    Run focused experiments with all methods tested on a single, consistent dataset.
    """
    # Define objectives - each reports its own metric
    objectives = [
        ('Sharpe Ratio', sharpe_performance_function),
        ('Monotonicity', monotonicity_performance_function),
        ('Calmar Ratio', calmar_performance_function)
    ]

    # Updated set of methods including the new TwoLevelContextualBayesian
    methods_to_test = [
        ('Random Uniform', lambda: RandomUniformSampling()),
        ('Fast Hierarchical', lambda: FastHierarchicalSampling(equal_weight_bias=0.2)),
        ('Pure Bayesian', lambda: PureBayesianSampling(
            n_initial_random=15, kernel_type='matern52',
            acquisition_function='ei', alpha=1e-3
        )),
        ('Tree Level Bayesian', lambda: TreeLevelBayesian(
            equal_weight_bias=0.2, n_initial_random=20,
            acquisition_function='ei', kernel_type='matern52'
        )),
        ('Two-Level Contextual', lambda: TwoLevelContextualBayesian(
            cluster_split_level=0.6, min_cluster_size=3, max_cluster_size=7,
            n_initial_random=25, acquisition_function='ei', kernel_type='matern52'
        ))
    ]

    # Results storage
    all_results = {}

    print("=" * 100)
    print("FOCUSED PORTFOLIO OPTIMIZATION EXPERIMENTS")
    print("=" * 100)
    print(f"Testing {len(objectives)} objectives × {len(methods_to_test)} methods = {len(objectives) * len(methods_to_test)} total experiments")
    print()

    # =============================================================================
    # KEY CHANGE: Initialize the environment ONCE to ensure consistent data
    # =============================================================================
    print("Initializing a single TestEnvironment to ensure all experiments use the same data...")
    try:
        # Initialize with the first objective; it will be updated in the loop.
        env = TestEnvironment(
            n_assets=200,
            performance_function=objectives[0][1],
            use_real_data=True
        )
        print("✓ Data loaded successfully. All tests will run on this static dataset.")
    except Exception as e:
        print(f"❌ Failed to initialize TestEnvironment: {e}")
        return None
    # =============================================================================

    # Run experiments for each objective on the SAME environment
    for obj_name, obj_function in objectives:
        print(f"\n{'='*80}")
        print(f"OBJECTIVE: {obj_name.upper()}")
        print(f"Performance Metric: {obj_function.__name__}")
        print(f"{'='*80}")

        # KEY CHANGE: Set the objective for the shared environment and clear old results
        env.performance_function = obj_function
        env.results = {}  # Clear results from the previous objective's run

        # Create fresh sampler instances for this objective run
        samplers = [factory() for name, factory in methods_to_test]
        for sampler, (name, factory) in zip(samplers, methods_to_test):
            sampler._method_name = name  # Assign clean name for reporting

        print(f"Running {len(samplers)} methods for {obj_name}...")

        # Run all experiments for this objective
        for i, sampler in enumerate(samplers):
            method_name = getattr(sampler, '_method_name', type(sampler).__name__)
            print(f"\n[{i+1}/{len(samplers)}] Testing: {method_name}")

            try:
                results = env.run_experiment(
                    sampling_method=sampler,
                    time_limit_seconds=60,
                    patience=20000
                )
                print(f"✓ Completed: {method_name} - {obj_name} Score: {results['final_best_score']:.6f}")
            except Exception as e:
                print(f"❌ Failed: {method_name} - Error: {e}")
                continue

        # KEY CHANGE: Store a COPY of the results for this objective, not the whole env
        all_results[obj_name] = {
            'results_data': env.results.copy(),
            'performance_function': obj_function
        }

        # Pass the results dictionary directly to the summary function
        print_objective_summary(env.results, obj_name, obj_function)

    # Cross-objective analysis and plotting now use the modified `all_results` structure
    print_cross_objective_analysis(all_results)
    create_focused_plots(all_results)

    return all_results

def print_objective_summary(results_data, obj_name, obj_function):
    """
    Print summary for a single objective.
    MODIFIED to accept a dictionary of results instead of the full environment object.
    """
    print(f"\n{'-'*60}")
    print(f"{obj_name.upper()} RESULTS")
    print(f"Metric: {obj_function.__name__}")
    print(f"{'-'*60}")

    results_list = sorted(results_data.items(), key=lambda item: item[1]['final_best_score'], reverse=True)
    
    equal_weight_score = results_data.get('Equal Weight', {}).get('final_best_score')

    print(f"{'Method':<35} {'Score':<12} {'Iterations':<12} {'Converged':<10}")
    print("-" * 75)

    for name, res in results_list:
        converged_str = "Yes" if res.get('converged', False) else "No"
        print(f"{name:<35} {res['final_best_score']:<12.6f} {res['total_iterations']:<12} {converged_str:<10}")

    if equal_weight_score is not None:
        print(f"\nEqual Weight Baseline: {equal_weight_score:.6f}")
        print("\nImprovement over Equal Weight:")
        for name, res in results_list:
            if 'Equal Weight' not in name:
                improvement = res['final_best_score'] - equal_weight_score
                # Avoid division by zero if baseline is 0
                improvement_pct = (improvement / abs(equal_weight_score)) * 100 if abs(equal_weight_score) > 1e-9 else float('inf')
                print(f"  {name:<35} +{improvement:>8.4f} ({improvement_pct:>+6.1f}%)")

def print_cross_objective_analysis(all_results):
    """
    Print cross-objective analysis.
    MODIFIED to handle the new `all_results` structure.
    """
    print(f"\n{'='*100}")
    print("CROSS-OBJECTIVE ANALYSIS")
    print(f"{'='*100}")

    comparison_data = []
    for obj_name, data in all_results.items():
        for method_name, method_results in data['results_data'].items():
            comparison_data.append({
                'Objective': obj_name,
                'Method': method_name,
                'Score': method_results['final_best_score'],
                'Iterations': method_results['total_iterations'],
                'Converged': method_results.get('converged', False),
                'Metric_Function': data['performance_function'].__name__
            })

    if not comparison_data:
        print("No data to analyze.")
        return

    comparison_df = pd.DataFrame(comparison_data)

    # ... (Rest of the function logic is sound and remains the same)
    # Print top performer for each objective
    print("\nTOP PERFORMER BY OBJECTIVE:")
    print(f"{'Objective':<20} {'Method':<35} {'Score':<12} {'Metric':<20}")
    print("-" * 90)
    
    for obj_name in comparison_df['Objective'].unique():
        obj_data = comparison_df[comparison_df['Objective'] == obj_name]
        if not obj_data.empty:
            top_performer = obj_data.loc[obj_data['Score'].idxmax()]
            print(f"{top_performer['Objective']:<20} {top_performer['Method']:<35} "
                  f"{top_performer['Score']:<12.6f} {top_performer['Metric_Function']:<20}")

def create_focused_plots(all_results):
    """
    Create focused visualization plots.
    MODIFIED to handle the new `all_results` structure.
    """
    num_objectives = len(all_results)
    if num_objectives == 0:
        return

    # 1. Performance comparison bar charts
    fig, axes = plt.subplots(1, num_objectives, figsize=(7 * num_objectives, 6), sharey=False)
    # Ensure axes is always a list for consistent indexing
    if num_objectives == 1:
        axes = [axes]

    for i, (obj_name, data) in enumerate(all_results.items()):
        objective_results = data['results_data']
        sorted_results = sorted(objective_results.items(), key=lambda item: item[1]['final_best_score'], reverse=True)

        methods = [name.replace('Two-Level Contextual', '2L-Context').replace('Tree Level Bayesian', 'TreeBayes').replace('Pure Bayesian', 'PureBayes').replace('Fast Hierarchical', 'FastHier').replace('Random Uniform', 'Random') for name, res in sorted_results]
        scores = [res['final_best_score'] for name, res in sorted_results]

        ax = axes[i]
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(methods)))
        bars = ax.bar(methods, scores, color=colors)
        ax.set_title(f'{obj_name}\n({data["performance_function"].__name__})', fontsize=14)
        ax.set_ylabel('Final Score', fontsize=12)
        ax.tick_params(axis='x', rotation=45, ha='right')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, axis='y')

        # Add value labels on bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom', ha='center')

    plt.tight_layout(pad=2.0)
    plt.suptitle("Method Performance by Objective", fontsize=16, y=1.03)
    plt.show()

# Quick test and main execution block are omitted for brevity,
# but they should follow the same pattern of creating the environment once.
if __name__ == "__main__":
    print("FOCUSED PORTFOLIO OPTIMIZATION EXPERIMENTS")
    print("=" * 50)
    # ... (Introductory prints remain the same) ...
    
    print("OPTIONS:")
    print("1. Run quick test (Sharpe Ratio only, 10 seconds per run)")
    print("2. Run full focused experiments (3 objectives, 60 seconds per run)")
    print("3. Cancel")
    
    choice = input("\nEnter choice (1/2/3): ")
    
    if choice == '1':
        print("\n🧪 Running quick focused test...")
        # The quick test now implicitly uses the single-environment pattern
        run_quick_test()
    
    elif choice == '2':
        print("\n🚀 Starting full, focused experiments on a consistent dataset...")
        results = run_focused_experiments()
        if results:
            print(f"\n✅ All focused experiments completed!")
    else:
        print("Cancelled.")