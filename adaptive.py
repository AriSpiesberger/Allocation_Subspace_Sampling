# -*- coding: utf-8 -*-
"""
Focused Portfolio Optimization Experiments
Fixed to report correct metrics and reduced method set
Now includes TwoLevelContextualBayesian
@author: AriSpiesberger
"""
from src.sampling_methods import *
from src.objectives import *
from src.testing_improved import TestEnvironment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from src.testing_improved import TestEnvironment, AssetDataGenerator # MODIFIED: Import AssetDataGenerator
warnings.filterwarnings('ignore')

def run_focused_experiments():
    """
    Run focused experiments with a shared dataset for all objectives.
    """
    
    # Define objectives - each reports its own metric
    objectives = [
        ('Sharpe Ratio', sharpe_performance_function),
        ('Monotonicity', monotonicity_performance_function), 
        ('Calmar Ratio', calmar_performance_function)
    ]
    
    # Updated set of methods to test
    methods_to_test = [
        ('Random Uniform', lambda: RandomUniformSampling()),
        ('Fast Hierarchical', lambda: FastHierarchicalSampling(equal_weight_bias=0.2)),
        ('Pure Bayesian', lambda: PureBayesianSampling(
            n_initial_random=15,
            kernel_type='matern52',
            acquisition_function='ei',
            alpha=1e-3
        )),
        ('Tree Level Bayesian', lambda: TreeLevelBayesian(
            equal_weight_bias=0.2,
            n_initial_random=20,
            acquisition_function='ei',
            kernel_type='matern52'
        )),
        ('Two-Level Contextual', lambda: TwoLevelContextualBayesian(
            cluster_split_level=0.6,
            min_cluster_size=3,
            max_cluster_size=7,
            n_initial_random=25,
            acquisition_function='ei',
            kernel_type='matern52'
        ))
    ]
    
    # --- Fetch data ONCE before all experiments ---
    print("\nFetching shared asset data for all experiments...")
    n_assets_to_use = 200
    returns_data, asset_names, price_data = AssetDataGenerator.get_sp500_assets(n_assets_to_use)
    print(f"Data for {len(asset_names)} assets fetched. All experiments will use this dataset.")
    print("-" * 100)
    
    # Results storage
    all_results = {}
    
    print("="*100)
    print("FOCUSED PORTFOLIO OPTIMIZATION EXPERIMENTS")
    print("="*100)
    print(f"Testing {len(objectives)} objectives × {len(methods_to_test)} methods = {len(objectives) * len(methods_to_test)} total experiments")
    print()
    
    # Run experiments for each objective using the SAME data
    for obj_name, obj_function in objectives:
        print(f"\n{'='*80}")
        print(f"OBJECTIVE: {obj_name.upper()}")
        print(f"Performance Metric: {obj_function.__name__}")
        print(f"{'='*80}")
        
        # Create test environment using the SHARED data
        env = TestEnvironment(
            n_assets=len(asset_names),  # Use the actual number of assets fetched
            performance_function=obj_function,
            use_real_data=True,
            # Pass the pre-loaded data
            returns_data=returns_data,
            asset_names=asset_names,
            price_data=price_data
        )
        
        # Create samplers
        samplers = []
        for method_name, method_factory in methods_to_test:
            sampler = method_factory()
            sampler._method_name = method_name  # Store clean method name
            samplers.append(sampler)
        
        print(f"Running {len(samplers)} methods for {obj_name}...")
        
        # Run all experiments for this objective
        for i, sampler in enumerate(samplers):
            method_name = getattr(sampler, '_method_name', sampler.get_name())
            print(f"\n[{i+1}/{len(samplers)}] Testing: {method_name}")
            
            try:
                # Run individual experiment
                results = env.run_experiment(
                    sampling_method=sampler,
                    time_limit_seconds=60,
                    patience=20000
                )
                print(f"✓ Completed: {method_name} - {obj_name} Score: {results['final_best_score']:.6f}")
                
            except Exception as e:
                print(f"❌ Failed: {method_name} - Error: {e}")
                continue
        
        # Store results for this objective
        all_results[obj_name] = {
            'env': env,
            'samplers': samplers,
            'performance_function': obj_function
        }
        
        # Print summary for this objective
        print_objective_summary(env, obj_name, obj_function)
    
    # Perform final analysis across all objectives
    print_cross_objective_analysis(all_results)
    
    # Create final visualizations
    create_focused_plots(all_results)
    
    return all_results

def print_objective_summary(env, obj_name, obj_function):
    """Print summary for a single objective with correct metrics"""
    print(f"\n{'-'*60}")
    print(f"{obj_name.upper()} RESULTS")
    print(f"Metric: {obj_function.__name__}")
    print(f"{'-'*60}")
    
    # Get results and sort by performance
    results_list = [(name, res) for name, res in env.results.items()]
    results_list.sort(key=lambda x: x[1]['final_best_score'], reverse=True)
    
    # Find equal weight baseline performance
    equal_weight_score = None
    for name, res in results_list:
        if 'equal' in name.lower() or name == "Equal Weight":
            equal_weight_score = res['final_best_score']
            break
    
    print(f"{'Method':<35} {'Score':<12} {'Iterations':<12} {'Converged':<10}")
    print("-" * 75)
    
    for name, res in results_list:
        converged_str = "Yes" if res.get('converged', False) else "No"
        print(f"{name:<35} {res['final_best_score']:<12.6f} {res['total_iterations']:<12} {converged_str:<10}")
    
    if equal_weight_score is not None:
        print(f"\nEqual Weight Baseline: {equal_weight_score:.6f}")
        
        print(f"\nImprovement over Equal Weight:")
        for name, res in results_list:
            if 'equal' not in name.lower():
                improvement = res['final_best_score'] - equal_weight_score
                improvement_pct = (improvement / abs(equal_weight_score)) * 100
                print(f"  {name:<35} +{improvement:>8.4f} ({improvement_pct:>+6.1f}%)")

def print_cross_objective_analysis(all_results):
    """Print cross-objective analysis"""
    print(f"\n{'='*100}")
    print("CROSS-OBJECTIVE ANALYSIS")
    print(f"{'='*100}")
    
    # Create comparison dataframe
    comparison_data = []
    
    for obj_name, results in all_results.items():
        env = results['env']
        obj_function = results['performance_function']
        
        for method_name, method_results in env.results.items():
            comparison_data.append({
                'Objective': obj_name,
                'Method': method_name,
                'Score': method_results['final_best_score'],
                'Iterations': method_results['total_iterations'],
                'Converged': method_results['converged'],
                'Metric_Function': obj_function.__name__
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Print top performer for each objective
    print("\nTOP PERFORMER BY OBJECTIVE:")
    print(f"{'Objective':<20} {'Method':<35} {'Score':<12} {'Metric':<20}")
    print("-" * 90)
    
    for obj_name in comparison_df['Objective'].unique():
        obj_data = comparison_df[comparison_df['Objective'] == obj_name]
        top_performer = obj_data.loc[obj_data['Score'].idxmax()]
        
        print(f"{top_performer['Objective']:<20} {top_performer['Method']:<35} "
              f"{top_performer['Score']:<12.6f} {top_performer['Metric_Function']:<20}")
    
    # Method consistency across objectives
    print(f"\n{'='*80}")
    print("METHOD CONSISTENCY ANALYSIS")
    print(f"{'='*80}")
    
    # Calculate ranks for each method across objectives
    method_ranks = {}
    
    for obj_name in comparison_df['Objective'].unique():
        obj_data = comparison_df[comparison_df['Objective'] == obj_name]
        obj_data_sorted = obj_data.sort_values('Score', ascending=False)
        obj_data_sorted['Rank'] = range(1, len(obj_data_sorted) + 1)
        
        for idx, row in obj_data_sorted.iterrows():
            method = row['Method']
            if method not in method_ranks:
                method_ranks[method] = []
            method_ranks[method].append(row['Rank'])
    
    # Calculate average ranks
    avg_ranks = []
    for method, ranks in method_ranks.items():
        if len(ranks) == len(all_results):  # Only methods that ran on all objectives
            avg_rank = np.mean(ranks)
            min_rank = min(ranks)
            max_rank = max(ranks)
            consistency = max_rank - min_rank  # Lower is more consistent
            
            avg_ranks.append({
                'Method': method,
                'Avg_Rank': avg_rank,
                'Best_Rank': min_rank,
                'Worst_Rank': max_rank,
                'Consistency': consistency
            })
    
    # Sort by average rank
    avg_ranks.sort(key=lambda x: x['Avg_Rank'])
    
    print("\nMETHOD RANKING ACROSS ALL OBJECTIVES:")
    print(f"{'Method':<35} {'Avg Rank':<10} {'Best':<6} {'Worst':<6} {'Consistency':<12}")
    print("-" * 75)
    
    for result in avg_ranks:
        consistency_desc = "Excellent" if result['Consistency'] <= 1 else "Good" if result['Consistency'] <= 2 else "Variable"
        print(f"{result['Method']:<35} {result['Avg_Rank']:<10.1f} {result['Best_Rank']:<6.0f} "
              f"{result['Worst_Rank']:<6.0f} {consistency_desc:<12}")
    
    return comparison_df, avg_ranks

def create_focused_plots(all_results):
    """Create focused visualization plots"""
    
    # 1. Performance comparison across objectives
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    for i, (obj_name, results) in enumerate(all_results.items()):
        env = results['env']
        
        # Get results
        methods = []
        scores = []
        
        for method_name, method_results in env.results.items():
            # Shorten method names for display
            short_name = method_name.replace('Hierarchical Bayesian', 'HB') \
                                  .replace('Two-Level Contextual', '2Level') \
                                  .replace('Tree Level Bayesian', 'TreeBayes') \
                                  .replace('Pure Bayesian', 'PureBayes') \
                                  .replace('Fast Hierarchical', 'FastHier')
            methods.append(short_name)
            scores.append(method_results['final_best_score'])
        
        # Create bar plot with more colors
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
        bars = axes[i].bar(range(len(methods)), scores, 
                          color=colors[:len(methods)])
        
        axes[i].set_title(f'{obj_name}\n({results["performance_function"].__name__})')
        axes[i].set_xlabel('Method')
        axes[i].set_ylabel('Performance Score')
        axes[i].set_xticks(range(len(methods)))
        axes[i].set_xticklabels(methods, rotation=45, ha='right')
        axes[i].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(scores),
                        f'{score:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # 2. Convergence comparison
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
    
    for i, (obj_name, results) in enumerate(all_results.items()):
        env = results['env']
        
        for j, (method_name, method_results) in enumerate(env.results.items()):
            if j < len(colors):
                history = method_results['history']
                iterations = [h['iteration'] for h in history]
                best_scores = [h['best_score_so_far'] for h in history]
                
                short_name = method_name.replace('Hierarchical Bayesian', 'HB') \
                                      .replace('Two-Level Contextual', '2Level') \
                                      .replace('Tree Level Bayesian', 'TreeBayes') \
                                      .replace('Pure Bayesian', 'PureBayes') \
                                      .replace('Fast Hierarchical', 'FastHier')
                
                axes[i].plot(iterations, best_scores, label=short_name, 
                           linewidth=2, color=colors[j])
        
        axes[i].set_title(f'{obj_name} Convergence')
        axes[i].set_xlabel('Iteration')
        axes[i].set_ylabel('Best Score')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 3. Method comparison heatmap
    plt.figure(figsize=(14, 10))
    
    # Prepare data for heatmap
    objectives = list(all_results.keys())
    all_methods = set()
    for obj_name, results in all_results.items():
        all_methods.update(results['env'].results.keys())
    
    all_methods = sorted(list(all_methods))
    
    # Create performance matrix
    performance_matrix = np.full((len(all_methods), len(objectives)), np.nan)
    
    for i, method in enumerate(all_methods):
        for j, obj_name in enumerate(objectives):
            env = all_results[obj_name]['env']
            if method in env.results:
                performance_matrix[i, j] = env.results[method]['final_best_score']
    
    # Create heatmap
    plt.imshow(performance_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Performance Score')
    plt.xticks(range(len(objectives)), objectives)
    
    # Shorten method names for y-axis
    short_method_names = []
    for m in all_methods:
        short_name = m.replace('Hierarchical Bayesian', 'HB') \
                    .replace('Two-Level Contextual', '2Level') \
                    .replace('Tree Level Bayesian', 'TreeBayes') \
                    .replace('Pure Bayesian', 'PureBayes') \
                    .replace('Fast Hierarchical', 'FastHier')
        short_method_names.append(short_name)
    
    plt.yticks(range(len(all_methods)), short_method_names)
    plt.title('Method Performance Across Objectives')
    
    # Add text annotations
    for i in range(len(all_methods)):
        for j in range(len(objectives)):
            if not np.isnan(performance_matrix[i, j]):
                plt.text(j, i, f'{performance_matrix[i, j]:.3f}', 
                        ha='center', va='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def run_quick_test():
    """Run a quick test with the focused method set including TwoLevel"""
    print("🧪 RUNNING QUICK FOCUSED TEST WITH NEW TWO-LEVEL METHOD...")
    print("=" * 60)
    
    # Test with just Sharpe ratio
    env = TestEnvironment(
        n_assets=200,
        performance_function=sharpe_performance_function,
        use_real_data=True
    )
    
    # Test all 5 focused methods including the new one
    test_samplers = [
        ('Random Uniform', lambda: RandomUniformSampling()),
        ('Fast Hierarchical', lambda: FastHierarchicalSampling(equal_weight_bias=0.2)),
        ('Pure Bayesian', lambda: PureBayesianSampling(
            n_initial_random=15,
            kernel_type='matern52',
            acquisition_function='ei',
            alpha=1e-3
        )),
        ('Tree Level Bayesian', lambda: TreeLevelBayesian(
            equal_weight_bias=0.2,
            n_initial_random=20,
            acquisition_function='ei',
            kernel_type='matern52'
        )),
        ('Two-Level Contextual', lambda: TwoLevelContextualBayesian(
            cluster_split_level=0.6,
            min_cluster_size=3,
            max_cluster_size=7,
            n_initial_random=25,
            acquisition_function='ei',
            kernel_type='matern52'
        ))
    ]
    
    print(f"Testing {len(test_samplers)} focused methods including NEW Two-Level Contextual...")
    
    for i, (method_name, method_factory) in enumerate(test_samplers):
        print(f"\n[{i+1}/{len(test_samplers)}] Testing: {method_name}")
        
        try:
            sampler = method_factory()
            results = env.run_experiment(
                sampling_method=sampler,
                time_limit_seconds=10, # MODIFIED: Reduced for quick test
                patience=2000
            )
            print(f"✓ Success: {results['final_best_score']:.6f}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            print(f"Full error: {traceback.format_exc()}")
    
    # Print results
    print_objective_summary(env, "Sharpe Ratio (Quick Test)", sharpe_performance_function)
    
    print("\n✅ Quick test completed! Ready for full focused experiments.")
    return env

if __name__ == "__main__":
    print("FOCUSED PORTFOLIO OPTIMIZATION EXPERIMENTS")
    print("=" * 50)
    print("🎯 FOCUSED METHOD SET (UPDATED):")
    print("  • Random Uniform: Baseline random sampling")
    print("  • Fast Hierarchical: Asset clustering approach (pure hierarchical)")  
    print("  • Pure Bayesian: Direct Bayesian optimization over full vector")
    print("  • Tree Level Bayesian: Ultra-compressed tree level optimization")
    print("  • Two-Level Contextual: NEW! Context tree + cluster bandits")
    print()
    print("📊 OBJECTIVES (each reports its own metric):")
    print("  • Sharpe Ratio → sharpe_performance_function")
    print("  • Monotonicity → monotonicity_performance_function") 
    print("  • Calmar Ratio → calmar_performance_function")
    print()
    print("✨ NEW METHOD HIGHLIGHTS:")
    print("  🏗️ Two-level architecture: Upper context tree + Lower asset clusters")
    print("  🧠 Dual optimization: Context GP + Contextual bandits per cluster")
    print("  📊 Smart clustering: Size constraints + singleton detection")
    print("  🎯 Reduced complexity while maintaining hierarchical benefits")
    print()
    
    # Ask user what they want to do
    print("OPTIONS:")
    print("1. Run quick test (5 methods, Sharpe only, 10 seconds per run)")
    print("2. Run full focused experiments (5 methods × 3 objectives, 60 seconds per run)")
    print("3. Cancel")
    
    choice = input("\nEnter choice (1/2/3): ")
    
    if choice == '1':
        print("\n🧪 Running quick focused test with NEW Two-Level method...")
        test_env = run_quick_test()
    
    elif choice == '2':
        print("\n🚀 Starting focused experiments with NEW Two-Level method...")
        all_results = run_focused_experiments()
        
        print(f"\n✅ All focused experiments completed!")
        print(f"📋 Results available in: all_results")
        print(f"🆕 Two-Level Contextual method results included!")
        
    else:
        print("Cancelled.")