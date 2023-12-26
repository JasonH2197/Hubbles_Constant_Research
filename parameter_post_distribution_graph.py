import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def graph_posterior_distribution(json_file_path, parameter_name, num_bins=50):
    
    """
    This function parses and extracts the values of a specified parameter 
    from the JSON file. Then, calculates the best-fit Gaussian curve for
    the histogram of the parameter's posterior distribution and plots
    this histogram overlayed with the  best-fit Gaussian curve, 
    highlighting the injection value and estimated mean.
    The resulting plot is saved as an image file and displayed.
    
    INPUT:
    -----
    data(str): .json data containing posterior and injection parameters.
    parameter_name(str): The name of the parameter for which the posterior distribution will be plotted.
    num_bins(int, optional): Number of bins for the histogram. Default is 50.

    EXAMPLE
    -------
    graph_posterior_distribution('path/to/your/data.json', luminosity_distance)
    """
    with open(json_file_path, "r") as f:
         data = json.load(f)

    # Extracting the posterior values and the injection value for the specified parameter
    posterior = data['posterior']
    content = posterior['content']
    parameter_values = np.array(content[parameter_name])

    injection_parameter = data['injection_parameters']
    injection_value = injection_parameter[parameter_name]

    # Calculate the best fit function for the posterior distribution histogram
    freq, bin_edge = np.histogram(parameter_values, bins=num_bins, density=True)
    bin_center = (bin_edge[1:] + bin_edge[:-1]) * 0.5

    def normal_pdf(x, mu, sigma):
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    initial_guess = [np.mean(parameter_values), np.std(parameter_values)]
    params, params_covariance = curve_fit(normal_pdf, bin_center, freq, p0=initial_guess)
    mean_estimate, std_dev_estimate = params

    x = np.linspace(np.min(parameter_values), np.max(parameter_values), 500)
    y = normal_pdf(x, mean_estimate, std_dev_estimate)
   
    # Plot the parameter posterior distribution as a histogram
    plt.hist(parameter_values, edgecolor='black', bins=num_bins, color='skyblue', alpha=0.6, density=True)
    plt.plot(x, y, color='blue', linewidth=2)
    plt.axvline(injection_value, color='red', linestyle='dashed', linewidth=2, label=f'{parameter_name.replace("_", " ").title()} injection: {injection_value}')
    plt.axvline(x=mean_estimate, color='orange', linestyle='-', linewidth=2, label=f'Estimated Mean: {mean_estimate:.2f}')
    plt.xlabel(parameter_name.replace('_', ' ').title())
    plt.ylabel('Frequency')
    plt.title(f'{parameter_name.replace("_", " ").title()} Posterior Distribution')
    plt.legend()
    plt.savefig(f'{parameter_name}_posterior.png')
    plt.show()

