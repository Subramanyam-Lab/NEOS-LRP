from deephyper.problem import HpProblem
from deephyper.evaluator import Evaluator
from deephyper.search.hps import CBO
from objective import run
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
from train_test_val import master
import os
import argparse

name_pattern= None
storage_path_csvs = "training/codes"

parser = argparse.ArgumentParser()
parser.add_argument('--surrogate_model', type=str, default='GP', help='Type of surrogate model to use')
args = parser.parse_args()

surrogate_model = args.surrogate_model

Problem = HpProblem()

Problem.add_hyperparameter([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], "latent_space_dimension")
Problem.add_hyperparameter([2, 3, 4, 5], "num_layers_phi")
Problem.add_hyperparameter([2, 3, 4, 5], "num_layers_rho")
Problem.add_hyperparameter([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], "neurons_per_layer_phi")
Problem.add_hyperparameter([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], "neurons_per_layer_rho")
Problem.add_hyperparameter(['Adam'], "optimizer")
Problem.add_hyperparameter(['MSE'], "loss_function")
Problem.add_hyperparameter(['relu'], "activation_function")
Problem.add_hyperparameter([10, 15, 20], "early_stopping_patience")
Problem.add_hyperparameter([32, 64, 128, 256, 512, 1024], "batch_size")
Problem.add_hyperparameter([0.01, 0.001, 0.0001], "learning_rate")
Problem.add_hyperparameter([50, 100, 150], "num_epochs")


if __name__ == "__main__":
    
    multiprocessing.set_start_method('spawn')

    filename_suffix = f"_{name_pattern}" if name_pattern else ""
    csv_file_name = os.path.join(storage_path_csvs, f'best_config_{surrogate_model}{filename_suffix}.csv')
    
    if os.path.isfile(csv_file_name):
        df = pd.read_csv(csv_file_name)
        best_row = df.loc[df['objective'].idxmin()]        
        initial_points = [best_row.drop('objective').to_dict()]
    else:
        initial_points = []
    evaluator = Evaluator.create(run, method="process", method_kwargs={ "num_workers": 1,},)

    search = CBO(problem=Problem, evaluator=evaluator,random_state=42, surrogate_model=surrogate_model)

    results = search.search(max_evals=50)
    
    results['objective'] = pd.to_numeric(results['objective'], errors='coerce')

    results = results.dropna(subset=['objective'])

    i_max = (results.objective.argmax())
    
    best_config = results.iloc[i_max][:-3].to_dict()

    best_config = {key.replace('p:', ''): value for key, value in best_config.items()}

    print("Best Configuration:")

    print(best_config)

    results['best_objective'] = -results['objective'].cummax()

    best_config['objective'] = results['best_objective'].iloc[-1]

    plt.figure(figsize=(10,5))
    plt.plot(results.index, results['best_objective'])
    plt.xlabel('Evaluation')
    plt.ylabel('Best Avg Test Loss (Cost)')
    # plt.savefig('BestObjvsE.png')
    plot_file_name = os.path.join(storage_path_csvs, f'BestObjvsE{filename_suffix}.png')
    plt.savefig(plot_file_name)

    metrics = master(best_config, metrics=True, exportonnx=True, testing=True, seed=42, N_dim=3)

    desired_metrics = ['test_cost_loss', 'test_opt_gap_cost_percent', 'test_opt_gap_cost_abs_percent','avg_new_metrics_abs','avg_new_metrics_abs']
    for metric in desired_metrics:
        if metric in metrics:
            value = metrics[metric].item() if hasattr(metrics[metric], 'item') else metrics[metric]
            best_config[metric] = value

    best_config_df = pd.DataFrame([best_config])
    if not os.path.isfile(csv_file_name):
        best_config_df.to_csv(csv_file_name, index=False)
    else:
        best_config_df.to_csv(csv_file_name, mode='a', header=False, index=False)

    for key, value in metrics.items():
        print(f'{key}: {value}\n')
        
