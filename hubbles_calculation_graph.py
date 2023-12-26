import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy import units
from astropy.cosmology import z_at_value, Planck18
from astropy import constants
import json

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

def plotting_hubbles_constant(json_file_path, num_bins=50):

    """
    Parses a .json file contaiing astrophysical data and generates a side-by-side
    histogram and best-fit normal distribution plot for luminosity distance and
    Hubble's constant. The plot showcases the injection values and estimated means
    based on the provided data. The resulting plot is then displayed.

    INPUT
    -----
    - data(str): Path to the JSON file containing luminosity distance and injection parameters.
    - num_bins(int): number of bins for the histograms. Default is 50.

    EXAMPLE
    -------
    plotting_hubbles_constant('path/to/your/data.json')
    """

    with open(json_file_path, "r") as f:
        data = json.load(f)

    # Extracts the luminosity distance data from a provided JSON file, computes
    # a histogram, and fits a normal distribution to the data. The resulting
    # histogram and best-fit normal distribution are then plotted on the first
    # subplot of a side-by-side visualization
    posterior = data['posterior']
    content = posterior['content']
    luminosity_distance = np.array(content['luminosity_distance'])

    injection_parameter = data['injection_parameters']
    luminosity_distance_inj = injection_parameter['luminosity_distance']

    hist, bin_edges = np.histogram(luminosity_distance, bins=num_bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    def normal_pdf(x, mu, sigma):
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    initial_guess = [np.mean(luminosity_distance), np.std(luminosity_distance)]
    params, params_covariance = curve_fit(normal_pdf, bin_centers, hist, p0=initial_guess)
    mean_estimate, std_dev_estimate = params

    x = np.linspace(np.min(luminosity_distance), np.max(luminosity_distance), 500)
    y = normal_pdf(x, mean_estimate, std_dev_estimate)

    axes[0].hist(luminosity_distance, edgecolor='black', bins=num_bins, color='skyblue', alpha=0.6, density=True)
    axes[0].plot(x, y, color='blue')
    axes[0].axvline(luminosity_distance_inj, color='red', linestyle='--', linewidth=2, label=f'Luminosity distance injection: {luminosity_distance_inj}')
    axes[0].axvline(x=mean_estimate, color='orange', linestyle='-', linewidth=2, label=f'Estimated Mean: {mean_estimate:.2f}')
    axes[0].set_xlabel('Luminosity Distance')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Histogram and Best-fit Normal Distribution')
    axes[0].legend()

    # Utilizes the luminosity distance injection value to calculate the corresponding
    # redshift (z) and, subsequently, the Hubble's constant. A histogram of the
    # Hubble's constant values is then created, and a normal distribution is fitted to
    # this data. The second subplot of the side-by-side visualization displays the
    # histogram and best-fit normal distribution for Hubble's constant.
    z = z_at_value(Planck18.luminosity_distance, float(luminosity_distance_inj) * units.Mpc).value
    speed_of_light = constants.c.to(units.km / units.s).value
    hubble_value = np.array((z * speed_of_light) / luminosity_distance)

    hist_H, bin_edges_H = np.histogram(hubble_value, bins=num_bins, density=True)
    bin_centers_H = 0.5 * (bin_edges_H[:-1] + bin_edges_H[1:])

    def normal_pdf(x, mu, sigma):
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    initial_guess_H = [np.mean(hubble_value), np.std(hubble_value)]
    params_H, params_covariance_H = curve_fit(normal_pdf, bin_centers_H, hist_H, p0=initial_guess_H)
    mean_estimate_H, std_dev_estimate_H = params_H

    x_H = np.linspace(np.min(hubble_value), np.max(hubble_value), 500)
    y_H = normal_pdf(x_H, mean_estimate_H, std_dev_estimate_H)

    axes[1].hist(hubble_value, edgecolor='black', bins=num_bins, color='skyblue', alpha=0.6, density=True)
    axes[1].plot(x_H, y_H, color='blue', linestyle='-')
    axes[1].axvline(x=mean_estimate_H, color='orange', linestyle='-', linewidth=2, label=f'Estimated Mean: {mean_estimate_H:.2f}')
    axes[1].set_xlabel("Hubble's constant")
    axes[1].set_ylabel('Frequency')
    axes[1].set_title("Histogram and Best-fit for Hubbles constant")
    axes[1].legend()

    # Displaying the graph with the two subplots
    plt.legend()
    plt.show()


