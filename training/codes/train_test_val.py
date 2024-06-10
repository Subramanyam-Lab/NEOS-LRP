import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from model_iisc import DeepSetArchitecture
from datapreprocessing_iisc import preprocess_data
import torch.nn as nn
import torch.optim as optim
import torch.onnx
import numpy as np
import time
import os
import random
import argparse
import csv

name_pattern = None
num_instances = int(os.getenv('num_instances'))

file_path = "/path/to/train/data"

X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(file_path, num_instances=num_instances, seed=42)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_model_statistics(model):
    total_neurons = sum(layer.out_features for layer in model.modules() if isinstance(layer, nn.Linear))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_weights = sum(p.numel() for p in model.parameters() if len(p.shape) > 1)
    total_biases = sum(p.numel() for p in model.parameters() if len(p.shape) == 1)
    total_non_zero_weights = sum(p.count_nonzero() for p in model.parameters() if len(p.shape) > 1)
    total_non_zero_biases = sum(p.count_nonzero() for p in model.parameters() if len(p.shape) == 1)

    return total_neurons, total_trainable_params, total_weights, total_biases, total_non_zero_weights, total_non_zero_biases


def master(config, metrics=True, exportonnx=True, testing=True, seed=42, N_dim=3):

    set_seed(seed)

    batch_size = int(config["batch_size"])
    num_epochs = int(config["num_epochs"])
    learning_rate=config["learning_rate"]

    # batch_size =32
    # learning_rate=0.0001
    # num_epochs=150

    #l1_lambda = config["l1_lambda"]
    #l2_lambda = config["l2_lambda"]

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda:0" if torch.backends.cuda.is_built() and torch.cuda.is_available() else "cpu")

    print("Running on:", device)
    print("Device status:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    model = DeepSetArchitecture(N_dim, config)
    model.to(device)

    criterion_cost = nn.MSELoss(reduction='mean')

    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif config['optimizer'] == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    
    best_loss = float('inf')
    early_stopping_patience=config["early_stopping_patience"]
    early_stopping_counter = 0

    train_losses_cost = []
    val_losses_cost = []

    train_val_start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_loss_cost = 0.0
        total_instances_train = 0  # no of instances in the train

        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs_cost = model(inputs)

            targets_cost = targets[:, 0]

            loss_cost = criterion_cost(outputs_cost, targets_cost)

            #l1_norm = sum(p.abs().sum() for p in model.parameters())
            #l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            
            #l1_loss = l1_lambda * l1_norm
            #l2_loss = l2_lambda * l2_norm

            total_loss = loss_cost
            
            total_loss.backward()
            
            optimizer.step()

            batch_size = inputs.shape[0]
            epoch_train_loss_cost += loss_cost.item() * batch_size
            epoch_train_loss += total_loss.item() * batch_size

            total_instances_train += batch_size

        epoch_train_loss_cost /= total_instances_train
        epoch_train_loss /= total_instances_train
        
        train_losses_cost.append(epoch_train_loss_cost)


        # print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {epoch_train_loss}")

        # Validation loop
        with torch.no_grad():
            model.eval()
            epoch_val_loss = 0.0
            epoch_val_loss_cost = 0.0
            total_instances_val = 0  # no of instances in the validation data

            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs_cost = model(inputs)

                targets_cost = targets[:, 0]

                val_loss_cost = criterion_cost(outputs_cost, targets_cost)

                total_val_loss = val_loss_cost

                batch_size = inputs.shape[0]
                epoch_val_loss_cost += val_loss_cost.item() * batch_size
                epoch_val_loss += total_val_loss.item() * batch_size

                total_instances_val += batch_size

            epoch_val_loss_cost /= total_instances_val
            epoch_val_loss /= total_instances_val

            val_losses_cost.append(epoch_val_loss_cost)

            print(f"Epoch {epoch + 1}/{num_epochs} - Validation Loss: {epoch_val_loss}")

            last_epoch_state_dict = model.state_dict().copy()

            # Early stopping check
            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                early_stopping_counter = 0
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print("Early stopping triggered.")
                    break
    
    train_val_end_time = time.time()
    train_val_time_sec = train_val_end_time - train_val_start_time

    if early_stopping_counter < early_stopping_patience:
        print("Saving model from last epoch as early stopping was not triggered.")
        torch.save(last_epoch_state_dict, 'last_epoch_model.pth')

    if testing:

        file_timestamp = None
        if os.path.exists('best_model.pth'):
            file_timestamp = os.path.getmtime('best_model.pth')

        # Decide which model to load based on the file's last modification time
        if file_timestamp is not None and file_timestamp >= train_val_start_time:
            print("Loading best model from this run.")
            model.load_state_dict(torch.load('best_model.pth'))
        else:
            print("Loading model from last epoch as best model was not saved in this run or is outdated.")
            model.load_state_dict(torch.load('last_epoch_model.pth'))

        with torch.no_grad():
            test_start_time = time.time()
            model.eval()
            test_loss = 0.0
            test_loss_cost = 0.0
            total_instances_test = 0  # no of instances in the testing data
            test_opt_gap_cost_abs = 0.0
            test_opt_gap_cost = 0.0
            total_new_metrics = 0.0
            total_new_metrics_abs = 0.0

            actual_costs = []
            predicted_costs = []

            for inputs, targets in test_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs_cost = model(inputs)

                targets_cost= targets[:, 0]

                actual_costs.extend(targets_cost.cpu().numpy())
                predicted_costs.extend(outputs_cost.cpu().detach().numpy())

                test_loss_cost = criterion_cost(outputs_cost, targets_cost)

                total_test_loss = test_loss_cost

                batch_size = inputs.shape[0]

                test_loss_cost += test_loss_cost.item() * batch_size
                test_loss += total_test_loss.item() * batch_size

                total_instances_test += batch_size

                test_opt_gap_cost_abs += torch.abs((targets_cost - outputs_cost) / targets_cost).sum().item()

                test_opt_gap_cost += ((targets_cost - outputs_cost) / targets_cost).sum().item()

                new_metrics_batch = ((targets_cost) - (outputs_cost )) / (targets_cost)

                total_new_metrics += new_metrics_batch.sum().item()

                new_metrics_abs_batch = torch.abs((targets_cost) - (outputs_cost)) / (targets_cost)

                total_new_metrics_abs += new_metrics_abs_batch.sum().item()


            test_loss_cost /= total_instances_test
            test_loss /= total_instances_test

            test_opt_gap_cost_abs = (test_opt_gap_cost_abs / total_instances_test) * 100

            test_opt_gap_cost = (test_opt_gap_cost / total_instances_test) * 100

            avg_new_metrics = (total_new_metrics / total_instances_test) * 100
            avg_new_metrics_abs = (total_new_metrics_abs / total_instances_test) * 100

            test_end_time = time.time()
            test_time_sec = test_end_time - test_start_time

            min_costs = min(min(actual_costs), min(actual_costs))
            max_costs = max(max(actual_costs), max(predicted_costs))

            with open('actual_vs_predicted.csv', 'w', newline='') as csvfile:
                fieldnames = ['actual_costs', 'predicted_costs']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for a, b in zip(actual_costs, predicted_costs):
                    writer.writerow({'actual_costs': a, 'predicted_costs': b})

            # cost 
            plt.figure(figsize=(10, 6))
            plt.scatter(actual_costs, predicted_costs, alpha=0.5, color='red', label='Predicted')
            plt.plot([min_costs, max_costs], [min_costs, max_costs], color='blue', label='Perfect prediction')
            plt.xlim([min_costs, max_costs])
            plt.ylim([min_costs, max_costs])
            plt.title('Actual vs Predicted Costs')
            plt.xlabel('Actual Costs')
            plt.ylabel('Predicted Costs')
            plt.legend()
            plt.savefig('costs.png', dpi=300)

    else:
        print("Testing is disabled.")
        test_loss = None
        test_loss_cost = None
        actual_costs = []
        predicted_costs = []


    if exportonnx:
        phi_model = model.phi
        rho_model = model.rho
        
        dummy_input_phi = torch.randn(1, N_dim).to(device)  # dummy input for phi
        torch.onnx.export(phi_model, dummy_input_phi, "model_phi_dnn.onnx")
        torch.save(phi_model.state_dict(), "final_model_phi_dnn.pth")
        
        dummy_input_rho = torch.randn(1, config['latent_space_dimension']).to(device)  # dummy input for rho
        torch.onnx.export(rho_model, dummy_input_rho, "model_rho_dnn.onnx")
        torch.save(rho_model.state_dict(), "model_rho_dnn.pth")
        
    else:
        print("No ONNX Export")

    if metrics:
        l1_norm = 0.0
        for param in model.parameters():
            l1_norm += torch.sum(torch.abs(param))
        total_neurons, total_trainable_params, total_weights, total_biases, total_non_zero_weights, total_non_zero_biases = get_model_statistics(model)

        metrics = {
            'train_losses_cost': train_losses_cost,
            'val_losses_cost': val_losses_cost,
            'best_val_loss': best_loss,
            'test_cost_loss': test_loss_cost,
            'test_opt_gap_cost_abs_percent': test_opt_gap_cost_abs,
            'test_opt_gap_cost_percent': test_opt_gap_cost_abs,
            'avg_new_metrics':avg_new_metrics,
            'avg_new_metrics_abs':avg_new_metrics_abs,
            'L1 Norm': l1_norm.item(),
            'Total Neurons': total_neurons,
            'Total Trainable Params': total_trainable_params,
            'Total Weights': total_weights,
            'Total Biases': total_biases,
            'Total Non-zero Weights': total_non_zero_weights,
            'Total Non-zero Biases': total_non_zero_biases,
            'Training Time (s)': train_val_time_sec,
            'Testing Time (s)': test_time_sec
        }

        return metrics
    else:

        print('HPO searching stage')
        return best_loss
