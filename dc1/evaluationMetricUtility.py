from typing import List
import torch
import plotext
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from dc1.train_test import train_model, test_model, label_names
from pathlib import Path
from datetime import datetime


class EvaluationMettricsLogger:
    def __init__(self):
        # Initialize lists to store metrics for training and testing
        self.mean_losses_train: List[torch.Tensor] = []
        self.mean_losses_test: List[torch.Tensor] = []
        self.accuracies_train, self.precisions_train, self.recalls_train = [], [], []
        self.accuracies_test, self.precisions_test, self.recalls_test = [], [], []

        # Initialize list to store ROC AUC data
        self.roc_auc_data = []
        self.min_mean_loss_train = None
        self.min_mean_loss_train = None
        self.now_time = datetime.now()
        self.labels_distr_test = []

    def log_training_epochs(self, epoch: int, model, train_sampler, optimizer, loss_function, device, MCd: bool = False):
        if not MCd:
            # Log training metrics for each epoch
            train_losses, train_acc, train_prec, train_rec, labels_distr, datadis = train_model(model, train_sampler,optimizer, loss_function, device)
            self.accuracies_train.append(train_acc)
            self.precisions_train.append(train_prec)
            self.recalls_train.append(train_rec)

            mean_loss = sum(train_losses) / len(train_losses)
            self.mean_losses_train.append(mean_loss)

            verbose = True
            if verbose == True:
                self._print_metrics(epoch, mean_loss, train_acc, train_prec, train_rec, labels_distr, datadis, train=True)

    def log_testing_epochs(self, epoch: int, model, test_sampler, loss_function, device):
        # Log testing metrics for each epoch, including ROC AUC
        test_losses, test_acc, test_prec, test_rec, roc_auc_dict, labels_distr, datadis, overall_avg = test_model(model, test_sampler, loss_function,device)
        #print(f"ROC AUC dict for epoch {epoch + 1}: {roc_auc_dict}")
        self.roc_auc_data.append(roc_auc_dict)
        self.accuracies_test.append(test_acc)
        self.precisions_test.append(test_prec)
        self.recalls_test.append(test_rec)
        mean_loss = sum(test_losses) / len(test_losses)
        self.mean_losses_test.append(mean_loss)

        self.labels_distr_test.append(labels_distr)

        verbose = True
        if verbose == True:
            self._print_metrics(epoch, mean_loss, test_acc, test_prec, test_rec, labels_distr, datadis, train=False)

        return overall_avg

    def plot_training_testing_losses(self,MCd: bool = False):
        if not MCd:
            # Plot scatter of training and testing losses using plotext
            plotext.clf()
            plotext.scatter(self.mean_losses_train, label="Train Loss")
            plotext.scatter(self.mean_losses_test, label="Test Loss")
            plotext.title("Train and Test Loss")
            plotext.xticks([i for i in range(len(self.mean_losses_train) + 1)])
            plotext.show()

    def plot_loss_and_accuracy(self, epochs):
        # Plot training and testing losses and accuracies over epochs

        figure(figsize=(9, 10), dpi=80)
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)

        ax1.plot(range(1, 1 + epochs), [x.detach().cpu() for x in self.mean_losses_train], label="Train Loss", marker='o',
                 color="blue")
        ax1.plot(range(1, 1 + epochs), [x.detach().cpu() for x in self.mean_losses_test], label="Test Loss", marker='o',
                 color="red")

        ax1.set_title("Train and Test Loss Over Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend(loc='upper right')

        ax2.plot(range(1, 1 + epochs), self.accuracies_train, label="Train Accuracy", marker='o', color="blue")
        ax2.plot(range(1, 1 + epochs), self.accuracies_test, label="Test Accuracy", marker='o', color="red")

        ax2.set_title("Train and Test Accuracy Over Epochs")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Accuracy")
        ax2.legend(loc='upper right')

        # Save plots to artifacts folder
        fig.savefig(Path("artifacts") / f"session_{self.now_time.month:02}_{self.now_time.day:02}_{self.now_time.hour}_{self.now_time.minute:02}.png")


    def plot_roc_curve(self, epochs, train_m):
        # Plotting and saving ROC Curves
        roc_figure, roc_axes = plt.subplots(figsize=(10, 8))
        for _, label_name in train_m.label_names.items():
            epoch_auc_scores = [self.roc_auc_data[epoch][label_name] for epoch in range(epochs)]
            roc_axes.plot(range(1, epochs + 1), epoch_auc_scores, marker='o', label=label_name)

        roc_axes.set_xlabel('Epochs')
        roc_axes.set_ylabel('ROC AUC')
        roc_axes.set_title('ROC AUC per Class Over Epochs')
        roc_axes.legend(loc='lower right')

        # Save plots to artifacts folder
        roc_figure.savefig(Path("artifacts") / f"roc_curves_{self.now_time.month:02}_{self.now_time.day:02}_{self.now_time.hour}_{self.now_time.minute:02}.png")

    def _print_metrics(self, epoch, mean_loss, acc, prec, rec, l_distr, datadis, train):
        phase = "Training" if train else "Testing"
        print(f"\nEpoch {epoch + 1} {phase} done:\n"
              f"{phase} Metrics:\n"
              f"Loss: {mean_loss}\n"
              f"🎯 Accuracy: {acc}\n"
              f"🎯 Precision: {prec}\n"
              f"🎯 Recall: {rec}\n")
        verbose = True
        if verbose:
            self.output_model_distr(l_distr, True, True, True)
            print("DATA DISTRIBUTION: ", datadis)

    def save_model(self, model, epoch: int, batch_size):
        filename = f"model_weights/model_{self.now_time.month:02}_{self.now_time.day:02}_{self.now_time.hour}_{self.now_time.minute:02}.pt"  # Use .pt for PyTorch models

        min_mean_loss_train = min([x.item() for x in self.mean_losses_train]) if self.mean_losses_train else None
        min_mean_loss_test = min([x.item() for x in self.mean_losses_test]) if self.mean_losses_test else None

        model_info = {
            "state_dict": model.state_dict(),
            "n_epochs": epoch,
            "batch_size": batch_size,
            "mean_losses_train": [x.item() for x in self.mean_losses_train],  # Convert tensors to numbers
            "mean_losses_test": [x.item() for x in self.mean_losses_test],
            "min_mean_loss_train": min_mean_loss_train,  # Add minimum mean loss for training
            "min_mean_loss_test": min_mean_loss_test,  # Add minimum mean loss for testing
        }

        torch.save(model_info, filename)

    def output_model_distr(self, l_distr, counts, percentages, output_as_percentage):
        for i, count_list in enumerate(l_distr):
            total_labels_for_disease = sum(count_list)
            correct_labels = count_list[i]  # Number of correct labels for the disease
            accuracy_percentage = (
                                          correct_labels / total_labels_for_disease) * 100 if total_labels_for_disease > 0 else 0

            print(f'For disease {label_names[i]} (Disease {i}):')
            if percentages:
                # Print accuracy as a percentage
                print(f'- Accuracy: {accuracy_percentage:.2f}%')

            if counts:
                # Print the count of correct and incorrect labels as originally intended
                for j, count in enumerate(count_list):
                    if i == j:
                        print(f'- {count} correct labels')
                    else:
                        if output_as_percentage:
                            # Calculate and print the percentage
                            total_guesses_for_class = sum([l_distr[x][j] for x in range(len(l_distr))])
                            percentage = (count / total_guesses_for_class) * 100 if total_guesses_for_class > 0 else 0
                            print(f'- {percentage:.2f}% incorrect guessed disease {j} ({label_names[j]})')
                        else:
                            # Print count as before
                            print(f'- {count} incorrect guessed disease {j} ({label_names[j]})')
            print()

    # def perform_grid_search():
    #     epoch_range = [5, 10, 15]
    #     batch_size_range = [16, 32, 64]
    #     best_loss = float('inf')
    #     best_config = None
    #
    #     for epochs in epoch_range:
    #         for batch_size in batch_size_range:
    #             print(f"Testing configuration: epochs={epochs}, batch_size={batch_size}")
    #             # Assuming your args variable is still needed for other configurations
    #             args = argparse.Namespace(nb_epochs=epochs, batch_size=batch_size, balanced_batches=True)
    #             # Call your modified main function with the current combination
    #             # You might need to capture the performance metric from your main function to use here
    #             main(args, epochs, batch_size, activeloop=True)
    #             # Here, you'd capture the performance metric and update best_loss and best_config accordingly
    #
    #     print(f"Best configuration: epochs={best_config['epochs']}, batch_size={best_config['batch_size']} with loss {best_loss}")


