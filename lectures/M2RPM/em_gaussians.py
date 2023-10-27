import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import os
print(os.environ['PATH'])

# Code pour compiler les figures matplotlib en Latex. 

#from matplotlib import rc
#rc('text', usetex=True)
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 18})
#rc('text', usetex=True)
#rc('lines', linewidth=2)

rougeCEA = "#b81420"
vertCEA = "#73be4b"

def generate_sample_from_mixture(num_samples, means, variances, mixing_coefficients):
    """
    Generate a sample from a mixture of Gaussians given the means, the variances and the mixing coefficents (i.e. the alpha_j)
    """

    num_components = len(means)
    data = np.zeros(num_samples)

    for i in range(num_samples):
        # Randomly choose a Gaussian component according to mixing coefficients
        component = np.random.choice(np.arange(num_components), p=mixing_coefficients)
        data[i] = np.random.normal(means[component], np.sqrt(variances[component]))

    return data


def initialize_parameters(data, num_components):
    """
    Initialization of the parameters (the means, the variances and the mixing coefficients)
    """
    num_features = data.shape[1]
    means = [np.random.rand(num_features) for _ in range(num_components)]
    variances = [np.eye(num_features) for _ in range(num_components)]
    mixing_coefficients = np.ones(num_components) / num_components
    return means, variances, mixing_coefficients

def expectation_step(data, means, variances, mixing_coefficients):
    """
    -- E-step of the EM algorithm --

    Compute the cluster weights for each data point with the output: cluster_weights[i, j] = q_{\Theta_k}(Z = j | X = x_i)
    """
    num_components = len(means)
    num_samples = data.shape[0]
    cluster_weights = np.zeros((num_samples, num_components))

    for i in range(num_samples):
        likelihoods = [multivariate_normal.pdf(data[i], means[j], variances[j]) for j in range(num_components)]
        weighted_likelihoods = mixing_coefficients * np.array(likelihoods)
        cluster_weights[i] = weighted_likelihoods / np.sum(weighted_likelihoods)

    return cluster_weights

def maximization_step(data, cluster_weights):
    """
    --- M-step of the EM algorithm -- 

    Update the means, variances, and mixing coefficients with the formula present in page 7/39 of the lecture notes. 
    """
    num_components = cluster_weights.shape[1]
    num_samples, num_features = data.shape

    means = np.zeros((num_components, num_features))
    variances = [np.zeros((num_features, num_features)) for _ in range(num_components)]
    mixing_coefficients = np.zeros(num_components)

    for j in range(num_components):
        total_responsibility = np.sum(cluster_weights[:, j])
        mixing_coefficients[j] = total_responsibility / num_samples

        weighted_data_sum = np.sum(data * cluster_weights[:, j].reshape(-1, 1), axis=0)
        means[j] = weighted_data_sum / total_responsibility

        diff_from_mean = data - means[j]
        variances[j] = np.dot(diff_from_mean.T, diff_from_mean * cluster_weights[:, j].reshape(-1, 1)) / total_responsibility

    return means, variances, mixing_coefficients

def log_likelihood(data, means, variances, mixing_coefficients):
    """
    Compute the log-likelihood of the mixture of Gaussians (Formula page 3/39 of the lecture notes)
    """
    num_components = len(means)
    num_samples = data.shape[0]
    likelihoods = np.zeros((num_samples, num_components))

    for i in range(num_samples):
        for j in range(num_components):
            likelihoods[i, j] = mixing_coefficients[j] * multivariate_normal.pdf(data[i], means[j], variances[j])

    return np.sum(np.log(np.sum(likelihoods, axis=1)))

def expectation_maximization(data, num_components, max_iterations=200, tolerance=1e-20):
    """
    EM algorithm, the parameters are stored in array during the computation, the number of iterations is fixed with max_iterations, it is possible to work with a tolerance parameter.
    """
    means, variances, mixing_coefficients = initialize_parameters(data, num_components)
    prev_log_likelihood = -np.inf
    log_likelihoods = []  # List to store log likelihood values
    means_ = [] 
    variances_ = []
    mixing_coefficients_ = []

    means_.append(means)
    variances_.append(variances)
    mixing_coefficients_.append(mixing_coefficients)

    for i in range(max_iterations):
        cluster_weights = expectation_step(data, means, variances, mixing_coefficients)
        means, variances, mixing_coefficients = maximization_step(data, cluster_weights)

        current_log_likelihood = log_likelihood(data, means, variances, mixing_coefficients)
        log_likelihoods.append(current_log_likelihood)  # Store the log likelihood
        means_.append(means)
        variances_.append(variances)
        mixing_coefficients_.append(mixing_coefficients)

        #if np.abs(current_log_likelihood - prev_log_likelihood) < tolerance:
        #    break

        #prev_log_likelihood = current_log_likelihood

    return means_, variances_, mixing_coefficients_, log_likelihoods  # Return log likelihoods


# Example usage:
np.random.seed(42)
data = generate_sample_from_mixture(1000, [1, 2, 3, 4], [0.2**2, 0.1**2, 0.25**2, 0.05**2], [0.2, 0.3, 0.4, 0.1])  # Sample data of 1000 points with 2 features
data = data[:, np.newaxis]

num_components = 4
estimated_means, estimated_variances, estimated_mixing_coefficients, log_likelihoods = expectation_maximization(data, num_components)

plt.plot(range(len(log_likelihoods)), log_likelihoods, 'b-')
plt.xlabel(r"Itérations")
plt.title(r"Log vraisemblance")
plt.tight_layout()
#plt.savefig("./loglik.pdf")
plt.show()

for i in range(4):
    estimated_means = np.array(estimated_means)
    plt.plot(range(201), estimated_means[:, i, 0], '-')
plt.xlabel(r"Itérations")
plt.title(r"Moyennes")
plt.tight_layout()
#plt.savefig("./means.pdf")
plt.show()

estimated_variances = np.sqrt(np.array(estimated_variances))

for i in range(4):
    plt.plot(range(201), estimated_variances[:, i, 0], '-')
plt.xlabel(r"Itérations")
plt.title(r"\'Ecarts type")
plt.tight_layout()
#plt.savefig("./std.pdf")
plt.show()

for i in range(4):
    estimated_mixing_coefficients = np.array(estimated_mixing_coefficients)
    plt.plot(range(201), estimated_mixing_coefficients[:, i], '-')
plt.xlabel(r"Itérations")
plt.title(r"Proportions")
plt.tight_layout()
#plt.savefig("./prop.pdf")
plt.show()