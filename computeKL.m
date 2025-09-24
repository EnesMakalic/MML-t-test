%COMPUTE_KL Computes the Kullback-Leibler divergence between two
%univariate Gaussians.
%
% Code returns
% KL(p || q), where p ~ N(mu1,sd1^2) and q ~ N(mu2, sd2^2)
%
function kl_divergence = computeKL(mu1, sd1, mu2, sd2)
arguments
        mu1 (1,1) double {mustBeNumeric,mustBeReal}
        sd1 (1,1) double {mustBeNumeric,mustBeReal, mustBePositive}
        mu2 (1,1) double {mustBeNumeric,mustBeReal}
        sd2 (1,1) double {mustBeNumeric,mustBeReal, mustBePositive}
end
    % Compute the KL divergence between two normal distributions
    % N(mu1, sd1^2) and N(mu2, sd2^2)

    % Calculate the KL divergence
    kl_divergence = log(sd2/sd1) + (sd1^2 + (mu1 - mu2)^2) / (2 * sd2^2) - 0.5;
end
