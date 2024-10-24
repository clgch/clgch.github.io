data {
    int<lower=0> N;  // Number of observations
    int<lower=0> K;  // Number of mixture components
    array[N] int<lower=0> y;  // Observed counts
    // à compléter 
}

parameters {
    // à compléter
}

model {
    // Priors
    for (k in 1:K) {
        // à compléter
    }

    // Likelihood
    for (n in 1:N) {
        real lambda_mix;
        lambda_mix = 0;
        for (k in 1:K) {
            // à compléter
        }
        y[n] ~ // à compléter;
    }
}