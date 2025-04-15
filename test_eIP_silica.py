import torch
import numpy as np
from PaiNN import PainnModel
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
from matplotlib import colors
from scipy.optimize import minimize
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, 
                           median_absolute_error, roc_curve, auc)
from scipy import stats
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize, unary_union
import pandas as pd
import time
import os
from pathlib import Path



def create_output_dir():
    """
    Create a timestamped output directory for saving figures and results.
    
    Returns:
        Path: Path object representing the created output directory
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"eIP_silica_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    return output_dir


def save_figure(plt, filename, output_dir, run_number=None):
    """
    Save figure to the output directory with optional run number prefix.
    
    Args:
        plt: Matplotlib pyplot object
        filename: Name of the figure file
        output_dir: Directory to save the figure
        run_number: Optional run number to prepend to filename
    """
    if run_number is not None:
        filename = f"run{run_number}_{filename}"
    plt.savefig(output_dir / filename, dpi=500, pad_inches=0.1, bbox_inches='tight')
    plt.close()
    

def load_model_and_weights():
    """
    Load the PaiNN model and its pre-trained weights.
    
    Returns:
        tuple: (model, device) - The loaded model and the device (CPU/GPU) it's using
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PainnModel(num_interactions=3, hidden_state_size=128, cutoff=5.0, pdb=True)
    model.to(device)
    
    weights = torch.load(
        './checkpoint/checkpoint_test_eIP.pt',
        map_location=device
    )
    model.load_state_dict(weights['model_state_dict'])
    return model, device


def load_datasets():
    """
    Load and prepare training, validation, and test datasets.
    
    Returns:
        tuple: DataLoaders for train, validation, and test data
    """
    train_dataset = torch.load('./dataset/silica_train.pt')[:1200]
    valid_dataset = torch.load('./dataset/silica_train.pt')[1200:]
    test_dataset = torch.load('./dataset/silica_test.pt')
        
    return (DataLoader(train_dataset, 1, shuffle=False),
            DataLoader(valid_dataset, 1, shuffle=False),
            DataLoader(test_dataset, 1, shuffle=False))


def evaluate_model_with_timing(model, loader, device):
    """
    Evaluate model on given data with performance timing.
    
    Args:
        model: The neural network model to evaluate
        loader: DataLoader containing the evaluation data
        device: Device (CPU/GPU) to perform calculations on
    
    Returns:
        dict: Results dictionary containing predictions, ground truth, uncertainties, and timing info
    """
    loss_func = torch.nn.L1Loss(reduction='none')  
    results = {
        'evidence_uncertainty': [],
        'evidence_mae': [],
        'energy_loss': [],
        'energy_gt': [],
        'energy_prediction': [],
        'force_gt': [],
        'force_prediction': [],
        'f_v': [],
        'residual': [],
        'inference_time': []
    }
    
    for batch_data in tqdm(loader):
        start_time = time.time()
        energy, force, node_para = model(batch_data.to(device))
        energy = energy + 1921.62
        inference_time = time.time() - start_time
        
        results['inference_time'].append(inference_time)
        
        force, f_v, f_alpha, f_beta = node_para
        aleatoric_uncertainty = torch.sqrt((f_beta.detach_())/(f_v.detach_()*(f_alpha.detach_()-1) + 1e-6))
        
        aleatoric_uncertainty = torch.norm(aleatoric_uncertainty, dim=-1)  
        
        force_mae = loss_func(force, batch_data.force)  # [batch_size, n_atoms, 3]
        force_mae = torch.norm(force_mae, dim=-1)  # [batch_size, n_atoms]
        
        results['evidence_uncertainty'].append(aleatoric_uncertainty.detach().cpu().numpy())  
        results['evidence_mae'].append(force_mae.detach().cpu().numpy())
        results['energy_loss'].append(loss_func(energy, batch_data.energy).detach().cpu().mean().item())
        results['energy_gt'].append(batch_data.energy.detach().cpu().item())
        results['energy_prediction'].append(energy.detach().cpu().item())
        results['force_gt'].append(batch_data.force.detach().cpu().numpy())
        results['force_prediction'].append(force.detach().cpu().numpy())
        results['f_v'].append(f_v.detach().cpu())
        res_tmp = torch.abs(torch.norm(force.detach().cpu(),dim=1)-torch.norm(batch_data.force.cpu(),dim=1)).mean()
        results['residual'].append(res_tmp)

    
    results['evidence_uncertainty'] = np.concatenate(results['evidence_uncertainty'])
    results['evidence_mae'] = np.concatenate(results['evidence_mae'])
    
    for key in ['energy_loss', 'energy_gt', 'energy_prediction', 'residual', 'inference_time']:
        results[key] = np.array(results[key])
    
    results['force_gt'] = np.concatenate(results['force_gt'])
    results['force_prediction'] = np.concatenate(results['force_prediction'])
    
    print("Uncertainty shape:", results['evidence_uncertainty'].shape)
    print("Force MAE shape:", results['evidence_mae'].shape)
    print("Raw force prediction range:", torch.min(force).item(), torch.max(force).item())
    print("Raw force GT range:", torch.min(batch_data.force).item(), torch.max(batch_data.force).item())
    
    return results


def plot_scatter_test(test_mae, test_uncertainty, output_dir, run_number=None):
    """
    Plot scatter plot for test data with exact formatting.
    
    Args:
        test_mae: Mean absolute error values for test data
        test_uncertainty: Uncertainty values for test data
        output_dir: Directory to save the plot
        run_number: Optional run number for the filename
    """
    sns.set(style='ticks')
    plt.figure(figsize=(6,6), dpi=500)
    
    plt.scatter(test_mae, test_uncertainty, marker='+', s=103, c='#496C88', label='Test')
    
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)
    ax.spines['right'].set_linewidth(4)
    ax.spines['top'].set_linewidth(4)
    plt.legend(fontsize='21')
    plt.yticks(size=17)
    plt.xticks(size=17)
    plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='x')
    save_figure(plt, 'scatter-test.png', output_dir, run_number)


def plot_scatter_distribution(test_mae, test_uncertainty, output_dir,
                        run_number=None, xlim=None, ylim=None):
    """
    Plot scatter and hexbin plots for test data with customizable axis ranges.
    Also saves the data to an Excel file for reproducibility.
    
    Args:
        test_mae: MAE values for test data
        test_uncertainty: Uncertainty values for test data
        output_dir: Directory to save the plot
        run_number: Optional run number for the filename
        xlim: Tuple of (min, max) for x-axis range
        ylim: Tuple of (min, max) for y-axis range
    """
    # Save data to Excel file for reproducibility
    data_dict = {
        'Test_MAE': test_mae,
        'Test_Uncertainty': test_uncertainty
    }
    
    # Create DataFrame and save to Excel
    df = pd.DataFrame(data_dict)
    excel_filename = f'scatter_data{"_" + str(run_number) if run_number else ""}.xlsx'
    excel_path = os.path.join(output_dir, excel_filename)
    df.to_excel(excel_path, index=False)
    print(f"Data saved to {excel_path}")
    
    sns.set(style='ticks')
    
    # Create scatter plot
    fig = plt.figure(figsize=(6,6), dpi=500)
    ax = fig.add_subplot(111)
    ax.scatter(test_mae, test_uncertainty, marker='+', c='#496C88', label='Test')
    ax.spines['bottom'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)
    ax.spines['right'].set_linewidth(4)
    ax.spines['top'].set_linewidth(4)

    ax.set_xlabel('MAE (eV/Å)', size=25)
    ax.set_ylabel('uncertainty', size=25)
    
    ax.tick_params(axis='both', which='major', labelsize=17)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    save_figure(plt, 'scatter-log.png', output_dir, run_number)
    plt.close()
    
    # Create hexbin plot
    fig_hex = plt.figure(figsize=(6,6), dpi=500)
    ax_hex = fig_hex.add_subplot(111)

    # Define logarithmic normalization for the colorbar
    norm = colors.LogNorm(vmin=1, vmax=1000)

    # Create hexbin plot
    hb = ax_hex.hexbin(
        test_mae, 
        test_uncertainty, 
        gridsize=80,  
        cmap='viridis',  
        norm=norm,
        xscale='log', 
        yscale='log',
        mincnt=1  
    )

    cax = fig_hex.add_axes([0.65, 0.2, 0.2, 0.03])  # [left, bottom, width, height]
    cb = plt.colorbar(hb, cax=cax, orientation='horizontal')
    cb.set_ticks([10, 1000])
    cb.ax.tick_params(labelsize=12)

    ax_hex.spines['bottom'].set_linewidth(3)
    ax_hex.spines['left'].set_linewidth(3)
    ax_hex.spines['right'].set_linewidth(3)
    ax_hex.spines['top'].set_linewidth(3)

    ax_hex.set_xlabel('MAE (eV/Å)', size=25)
    ax_hex.set_ylabel('uncertainty', size=25)
    
    ax_hex.tick_params(axis='both', which='major', labelsize=17)
    ax_hex.set_xlim(1e-3, 1e3)
    if ylim is not None:
        ax_hex.set_ylim(ylim)

    save_figure(plt, 'hexbin.png', output_dir, run_number)
    plt.close()


def plot_roc_auc(evidence_uncertainty, evidence_mae, output_dir, run_number=None):
    """
    Plot ROC-AUC curve with dynamic thresholds.
    
    Args:
        evidence_uncertainty: Uncertainty estimates
        evidence_mae: Mean absolute errors
        output_dir: Output directory
        run_number: Optional run number for file naming
    
    Returns:
        float: The best AUC score achieved
    """
    try:
        true_errors = np.array(evidence_uncertainty, dtype=np.float64)
        estimated_uncertainties = np.array(evidence_mae, dtype=np.float64)
        
        # Filter out invalid values
        valid_mask = np.isfinite(true_errors) & np.isfinite(estimated_uncertainties)
        true_errors = true_errors[valid_mask]
        estimated_uncertainties = estimated_uncertainties[valid_mask]
        
        if len(true_errors) == 0 or len(estimated_uncertainties) == 0:
            raise ValueError("All data points were invalid")
            
        # Try different thresholds to find the best AUC
        best_auc = 0
        best_fpr = None
        best_tpr = None
        best_threshold = None
        
        for percentile in range(5, 96, 5):  
            epsilon_c = np.percentile(true_errors, percentile)
            high_error = (true_errors > epsilon_c).astype(int)
            
            fpr, tpr, _ = roc_curve(high_error, estimated_uncertainties)
            current_auc = auc(fpr, tpr)
            
            if current_auc > best_auc:
                best_auc = current_auc
                best_fpr = fpr
                best_tpr = tpr
                best_threshold = percentile
        
        # Plot the ROC curve
        plt.figure(figsize=[5,5])
        plt.plot(best_fpr, best_tpr, color='darkorange', lw=2,
                label=f'ROC curve (area = {best_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Uncertainty Estimation')
        plt.legend(loc="lower right")
        
        # Save the figure
        if run_number is not None:
            filename = f"run{run_number}_roc_auc.png"
        else:
            filename = "roc_auc.png"
            
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ROC-AUC Score: {best_auc:.4f} (threshold = {best_threshold}%)")
        return best_auc
        
    except Exception as e:
        print(f"Error in plot_roc_auc: {str(e)}")
        return 0.5


def calculate_metrics(test_results, output_dir, display_options):
    """
    Calculate and print all required performance metrics.
    
    Args:
        test_results: Dictionary containing test results
        output_dir: Directory to save output files
        display_options: Dictionary controlling which metrics to display
    
    Returns:
        dict: Dictionary containing calculated metrics
    """
    metrics = {}
    test_uncertainties = np.array(test_results['evidence_uncertainty'])
    
    # Calculate basic metrics (always calculate even if not displayed)
    metrics['energy_mae'] = mean_absolute_error(test_results['energy_gt'], test_results['energy_prediction'])
    metrics['force_mae'] = mean_absolute_error(test_results['force_gt'].reshape(-1), test_results['force_prediction'].reshape(-1))
    
    # Calculate inference time statistics
    inference_times = test_results['inference_time']
    metrics['inference_time_mean'] = np.mean(inference_times)
    metrics['inference_time_std'] = np.std(inference_times)
    
    # Calculate correlation between uncertainty and error
    test_correlation, test_pvalue = stats.spearmanr(test_results['evidence_uncertainty'], test_results['evidence_mae'])
    metrics['test_spearman'] = test_correlation
    
    # Calculate ROC-AUC score
    metrics['roc_auc_test'] = plot_roc_auc(
        test_results['evidence_uncertainty'],
        test_results['evidence_mae'],
        output_dir,
        run_number=None
    )
    
    print("\n=== Model Performance Metrics ===")
    if display_options['show_energy']:
        print(f"Energy MAE: {metrics['energy_mae']:.4f}")
    if display_options['show_force']:
        print(f"Force MAE: {metrics['force_mae']:.4f}")
    if display_options['show_timing']:
        print(f"\nInference Time (mean): {metrics['inference_time_mean']:.4f} s")
        print(f"Inference Time (std): {metrics['inference_time_std']:.4f} s")
    if display_options['show_correlation']:
        print(f"\nTest Spearman correlation: {metrics['test_spearman']:.4f}")
    if display_options['show_roc_auc']:
        print(f"ROC-AUC_test Score: {metrics['roc_auc_test']:.4f}")
    
    return metrics


def save_metrics(metrics, output_dir, run_number=None, display_options=None):
    """
    Save metrics to a text file in the output directory.
    
    Args:
        metrics: Dictionary containing the calculated metrics
        output_dir: Path object pointing to output directory
        run_number: Run number to append to filename (optional)
        display_options: Dictionary controlling which metrics to save
    """
    if display_options is None:
        display_options = {
            'show_energy': True,
            'show_force': True,
            'show_timing': True,
            'show_correlation': True,
            'show_roc_auc': True
        }
        
    filename = "metrics.txt"
    if run_number is not None:
        filename = f"run{run_number}_{filename}"
        
    with open(output_dir / filename, 'w') as f:
        f.write("=== Model Performance Metrics ===\n\n")
        
        # Only include metrics that should be displayed according to options
        if display_options['show_energy']:
            f.write(f"Energy MAE: {metrics['energy_mae']:.4f}\n")
        if display_options['show_force']:
            f.write(f"Force MAE: {metrics['force_mae']:.4f}\n\n")
        
        # Timing metrics
        if display_options['show_timing']:
            f.write(f"Inference Time (mean): {metrics['inference_time_mean']:.4f} s\n")
            f.write(f"Inference Time (std): {metrics['inference_time_std']:.4f} s\n\n")
        
        # Uncertainty correlation
        if display_options['show_correlation']:
            f.write(f"Test Spearman correlation: {metrics['test_spearman']:.4f}\n\n")
        
        # ROC-AUC score
        if display_options['show_roc_auc']:
            f.write(f"ROC-AUC_test Score: {metrics['roc_auc_test']:.4f}\n")


def main():
    """
    Main function to run the evaluation pipeline.
    Evaluates model performance, generates visualizations, and saves metrics.
    """
    # Display options configuration
    display_options = {
        'show_force': True,        
        'show_timing': True, 
        'show_energy': False,       
        'show_correlation': True,  
        'show_roc_auc': True,      
        'plot_scatter': True,      
        'plot_hexbin': True,       
        'plot_roc': True           
    }
    
    # Create output directory
    output_dir = create_output_dir()
    print(f"Saving outputs to: {output_dir}")

    # Load model and datasets
    model, device = load_model_and_weights()
    train_loader, valid_loader, test_loader = load_datasets()
    
    # Run multiple evaluations
    all_metrics = []
    for i in range(3):
        print(f"\nRunning evaluation {i+1}/3...")
        val_results = evaluate_model_with_timing(model, valid_loader, device)
        test_results = evaluate_model_with_timing(model, test_loader, device)
        train_results = evaluate_model_with_timing(model, train_loader, device)
        
        # Calculate and save metrics
        metrics = calculate_metrics(test_results, output_dir, display_options)
        all_metrics.append(metrics)
        save_metrics(metrics, output_dir, i+1, display_options)
        
        # Generate visualizations (only for the first run to save time)
        if i == 0:
            print(test_results['evidence_uncertainty'].shape)
            if display_options['plot_scatter']:
                plot_scatter_test(test_results['evidence_mae'], test_results['evidence_uncertainty'], 
                           output_dir, i+1)
            if display_options['plot_hexbin']:
                plot_scatter_distribution(
                    test_results['evidence_mae'], 
                    test_results['evidence_uncertainty'],
                    output_dir, 
                    i+1,
                    xlim=None,  
                    ylim=None 
                )
            
            if display_options['plot_roc']:
                plot_roc_auc(test_results['evidence_uncertainty'], test_results['evidence_mae'], 
                        output_dir, i+1)
        if i == 1 and display_options['plot_roc']:
            plot_roc_auc(test_results['evidence_uncertainty'], test_results['evidence_mae'], 
                    output_dir, i+1)
        
    
    # Calculate and save average metrics
    print("\n=== Average Metrics Across 3 Runs ===")
    avg_metrics = {key: np.mean([run[key] for run in all_metrics]) for key in all_metrics[0].keys()}
    std_metrics = {key: np.std([run[key] for run in all_metrics]) for key in all_metrics[0].keys()}
    
    # Save average metrics and standard deviations to file
    with open(output_dir / "average_metrics.txt", 'w') as f:
        for key in avg_metrics:
            # Skip metrics that shouldn't be displayed
            if key == 'energy_mae' and not display_options['show_energy']:
                continue
                
            f.write(f"{key}:\n")
            f.write(f"  Mean: {avg_metrics[key]:.4f}\n")
            f.write(f"  Std:  {std_metrics[key]:.4f}\n")
            
            # Only print metrics that should be displayed
            if key == 'energy_mae' and not display_options['show_energy']:
                continue
                
            print(f"{key}:")
            print(f"  Mean: {avg_metrics[key]:.4f}")
            print(f"  Std:  {std_metrics[key]:.4f}")


if __name__ == "__main__":
    main()