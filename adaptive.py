# -*- coding: utf-8 -*-
"""
adaptive.py
REWRITTEN to be the central hub for running flexible, multi-objective, 
and multi-scenario portfolio optimization experiments. This version integrates 
all defined performance objectives for comprehensive testing.

@author: AriSpiesberger
"""

# Assuming src folder is in the same directory or accessible in the python path
from src.sampling_methods import *
from src.objectives import *
from src.testing_improved import TestEnvironment

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def run_comprehensive_experiments():
    """
    Runs a series of distinct experimental scenarios against a configurable
    set of performance objectives, all on a single, consistent dataset.
    """
    
    # =========================================================================
    # 1. DEFINE YOUR EXPERIMENTAL SCENARIOS
    #    Each scenario tests a different configuration of optimization methods.
    #    You can add or modify scenarios to test different hypotheses.
    # =========================================================================
    experimental_scenarios = {
        "Baseline": {
            "description": "Standard configuration with Matern kernel for Bayesian methods.",
            "methods_to_test": [
                ('Random Uniform', lambda: RandomUniformSampling()),
                ('Fast Hierarchical', lambda: FastHierarchicalSampling(equal_weight_bias=0.2)),
                ('Pure Bayesian (Matern)', lambda: PureBayesianSampling(n_initial_random=15, kernel_type='matern52')),
                ('Tree Level Bayesian', lambda: TreeLevelBayesian(equal_weight_bias=0.2, n_initial_random=20)),
                ('Two-Level Contextual', lambda: TwoLevelContextualBayesian(n_initial_random=25))
            ]
        },
        "Aggressive Exploration": {
            "description": "Bayesian methods use more initial random samples to explore the space.",
            "methods_to_test": [
                ('Random Uniform', lambda: RandomUniformSampling()),
                ('Fast Hierarchical', lambda: FastHierarchicalSampling(equal_weight_bias=0.2)),
                ('Pure Bayesian (Matern)', lambda: PureBayesianSampling(n_initial_random=40, kernel_type='matern52')),
                ('Tree Level Bayesian', lambda: TreeLevelBayesian(equal_weight_bias=0.2, n_initial_random=40)),
                ('Two-Level Contextual', lambda: TwoLevelContextualBayesian(n_initial_random=50))
            ]
        },
        "RBF Kernel": {
            "description": "Bayesian methods use the smoother RBF kernel instead of Matern.",
            "methods_to_test": [
                ('Random Uniform', lambda: RandomUniformSampling()),
                ('Fast Hierarchical', lambda: FastHierarchicalSampling(equal_weight_bias=0.2)),
                ('Pure Bayesian (RBF)', lambda: PureBayesianSampling(n_initial_random=15, kernel_type='rbf')),
                ('Tree Level Bayesian', lambda: TreeLevelBayesian(equal_weight_bias=0.2, n_initial_random=20, kernel_type='rbf')),
                ('Two-Level Contextual', lambda: TwoLevelContextualBayesian(n_initial_random=25, kernel_type='rbf'))
            ]
        }
    }

    # =========================================================================
    # 2. SELECT THE OBJECTIVES TO TEST
    #    Reference the PERFORMANCE_FUNCTIONS dictionary from your objectives.py file.
    #    You can test a subset or all of them.
    # =========================================================================
    objectives_to_test = [
        ('Sharpe Ratio', PERFORMANCE_FUNCTIONS['sharpe']),
        ('Sortino Ratio', PERFORMANCE_FUNCTIONS['sortino']),
        ('Calmar Ratio', PERFORMANCE_FUNCTIONS['calmar']),
        ('Comprehensive', PERFORMANCE_FUNCTIONS['comprehensive'])
        # Add more objectives here, e.g.: ('Omega Ratio', PERFORMANCE_FUNCTIONS['omega'])
    ]

    print("=" * 100)
    print("COMPREHENSIVE PORTFOLIO OPTIMIZATION EXPERIMENTS")
    print("=" * 100)
    print(f"Found {len(experimental_scenarios)} scenarios and {len(objectives_to_test)} objectives to run.")

    # Initialize the environment ONCE to ensure consistent data for all experiments
    print("\nInitializing a single TestEnvironment...")
    try:
        env = TestEnvironment(
            n_assets=200,
            performance_function=objectives_to_test[0][1], # Placeholder objective
            use_real_data=True
        )
        print("✓ Data loaded successfully. All tests will run on this static dataset.")
    except Exception as e:
        print(f"❌ Failed to initialize TestEnvironment: {e}")
        return

    # --- Main experimental loop ---
    for scenario_name, config in experimental_scenarios.items():
        print(f"\n\n{'#'*100}")
        print(f"RUNNING SCENARIO: {scenario_name.upper()}")
        print(f"DESCRIPTION: {config['description']}")
        print(f"{'#'*100}")

        scenario_results = {}
        methods_to_test = config['methods_to_test']

        # Run experiments for each objective within the current scenario
        for obj_name, obj_function in objectives_to_test:
            print(f"\n{'='*80}")
            print(f"OBJECTIVE: {obj_name.upper()} (under Scenario: {scenario_name})")
            print(f"{'='*80}")

            # Configure the shared environment for this specific run
            env.performance_function = obj_function
            env.results = {}  # Clear results from previous run

            samplers = [factory() for name, factory in methods_to_test]
            for sampler, (name, factory) in zip(samplers, methods_to_test):
                sampler._method_name = name

            # Run all experiments for this objective
            for sampler in samplers:
                method_name = getattr(sampler, '_method_name')
                print(f"\n--> Testing Method: {method_name}")
                try:
                    # Run the experiment
                    exp_results = env.run_experiment(
                        sampling_method=sampler,
                        time_limit_seconds=60, # Standard time limit per run
                        patience=20000
                    )
                    print(f"    ✓ Completed: Final Score = {exp_results['final_best_score']:.6f}")
                except Exception as e:
                    print(f"    ❌ Failed: {method_name} - Error: {e}")
                    continue
            
            # Store results for this objective within the current scenario
            scenario_results[obj_name] = {
                'results_data': env.results.copy(),
                'performance_function': obj_function
            }

        # --- Analysis and Visualization for the COMPLETED scenario ---
        print(f"\n\n{'-'*100}")
        print(f"ANALYSIS & RESULTS FOR SCENARIO: {scenario_name.upper()}")
        print(f"{'-'*100}")
        
        # Perform cross-objective analysis for the current scenario
        print_cross_objective_analysis(scenario_results, scenario_name)

        # Create plots for the current scenario
        create_scenario_plots(scenario_results, scenario_name)

    print("\n\n✅ All experimental scenarios completed.")


def print_cross_objective_analysis(scenario_results, scenario_name):
    """
    Prints a summary of the top-performing method for each objective in a scenario.
    """
    print(f"\n--- Top Performer by Objective for Scenario: '{scenario_name}' ---")
    print(f"{'Objective':<25} {'Method':<35} {'Score':<12}")
    print("-" * 75)

    for obj_name, data in scenario_results.items():
        results_data = data['results_data']
        if not results_data:
            continue
        
        # Find the best performing method for this objective
        top_method_name, top_method_results = max(results_data.items(), key=lambda item: item[1]['final_best_score'])
        
        print(f"{obj_name:<25} {top_method_name:<35} {top_method_results['final_best_score']:<12.6f}")


def create_scenario_plots(scenario_results, scenario_name):
    """
    Creates convergence plots for each objective within a completed scenario.
    """
    num_objectives = len(scenario_results)
    if num_objectives == 0:
        return

    # Create a subplot for each objective
    fig, axes = plt.subplots(1, num_objectives, figsize=(8 * num_objectives, 6), sharey=False)
    if num_objectives == 1: 
        axes = [axes] # Ensure axes is always a list

    for i, (obj_name, data) in enumerate(scenario_results.items()):
        ax = axes[i]
        results = data['results_data']
        
        # Plot convergence history for each method
        for method_name, method_results in results.items():
            history = method_results.get('history', [])
            if history:
                iterations = [h['iteration'] for h in history]
                best_scores = [h['best_score_so_far'] for h in history]
                ax.plot(iterations, best_scores, label=method_name, linewidth=2.5, alpha=0.8)
        
        ax.set_title(f'{obj_name} Convergence', fontsize=15)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Best Score Found', fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    fig.suptitle(f'Convergence Plots for Scenario: "{scenario_name}"', fontsize=20, y=1.03)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    # This is the single entry point to run all your experiments
    run_comprehensive_experiments()