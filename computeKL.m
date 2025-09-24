function kl_divergence = computeKL(mu1, sd1, mu2, sd2)
    % Compute the KL divergence between two normal distributions
    % N(mu1, sd1^2) and N(mu2, sd2^2)

    % Calculate the KL divergence
    kl_divergence = log(sd2/sd1) + (sd1^2 + (mu1 - mu2)^2) / (2 * sd2^2) - 0.5;
end
