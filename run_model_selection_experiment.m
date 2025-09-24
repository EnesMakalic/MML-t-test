% Experiment comparing BIC and MML87 model selection for Gaussian models.
% This code can be used to re-create Table 1 in the paper.
function results = run_model_selection_experiment(niters, opts)
arguments
    niters (1,1) double {mustBeInteger,mustBePositive} = 10
    opts.seed (1,1) double {mustBeInteger,mustBePositive} = 42
end

% For reproducibility
rng(opts.seed);  

% === Design grid ===
                               % sigma1 = 1;
sample_sizes = [15, 15];       % n1, n2
delta_vals = [0.5, 1.0, 2.0];  % effect size delta
variance_ratios = [0.5, 2, 4]; % sigma2
n_reps = niters;               % number of simulation runs

% === Storage ===
opts = optimoptions('fminunc','Display','off');
results = struct();

%% === Loop over generating models ===
for gen_model = 1:4
    fprintf('Generating from M%d...\n', gen_model);
    for delta = delta_vals
        for r = variance_ratios
            
            %% Set true parameters
            mu = 0;
            switch gen_model
                % model1: common mean and common s.d.         y1 ~ N(mu, sigma^2) and y2 ~ N(mu, sigma^2)
                case 1
                    sigma = 1;
                    params = struct('mu', mu, 'sigma', sigma);
                % model2: common mean and different s.d.      y1 ~ N(mu, sigma1^2) and y2 ~ N(mu, sigma2^2)
                case 2
                    sigma1 = 1;
                    sigma2 = r;
                    params = struct('mu', mu, 'sigma1', sigma1, 'sigma2', sigma2);
                % model3: different means and common s.d.     y1 ~ N(mu + (sigma/2)delta, sigma^2) and y2 ~ N(mu - (sigma/2)delta, sigma^2)
                case 3
                    sigma = 1;
                    params = struct('mu', mu, 'sigma', sigma, 'delta', delta);
                % model4: different means and different s.d.  y1 ~ N(mu + (sqrt(sigma1 sigma2)/2)delta, sigma1^2) and y2 ~ N(mu - (sqrt(sigma1 sigma2)/2)delta, sigma2^2)
                case 4
                    sigma1 = 1;
                    sigma2 = r;
                    params = struct('mu', mu, 'sigma1', sigma1, 'sigma2', sigma2, 'delta', delta);
            end

            % Run replicates for this selection of delta, r and generating
            % model
            acc_bic = zeros(n_reps,1);
            acc_mml = zeros(n_reps,1);
            kl_bic = zeros(n_reps,1);
            kl_mml = zeros(n_reps,1);

            % Do the simulation
            parfor rep = 1:n_reps
                % Simulate data
                [y1, y2] = simulate_data(gen_model, sample_sizes, params);

                %% Fit all models with maximum likelihood and compute BIC
                fits = cell(1,4);
                for m = 1:4
                    fits{m} = fit_model(m, y1, y2);
                end

                % Compute BIC criteria
                bic_vals = zeros(1,4);
                for m = 1:4
                    bic_vals(m) = compute_bic(fits{m}, sample_sizes);
                end

                %% Fit all models with MML87 and compute codelengths
                mmlfits = cell(1,4);
                [mml_vals, theta] = mmlttest(y1, y2, minoptions=opts);   
                id=1;
                mmlfits{id}.params.mu = theta{id}(1);
                mmlfits{id}.params.sigma = sqrt(theta{id}(2));
                mmlfits{id}.codelength = mml_vals(id);
                mmlfits{id}.model_id = id;

                id=2;
                mmlfits{id}.params.mu = theta{id}(1);
                mmlfits{id}.params.sigma1 = sqrt(theta{id}(2));
                mmlfits{id}.params.sigma2 = sqrt(theta{id}(3));
                mmlfits{id}.codelength = mml_vals(id);
                mmlfits{id}.model_id = id;      

                id=3;
                mmlfits{id}.params.mu = theta{id}(1);
                mmlfits{id}.params.sigma = sqrt(theta{id}(2));
                mmlfits{id}.params.delta = theta{id}(3);
                mmlfits{id}.codelength = mml_vals(id);
                mmlfits{id}.model_id = id;  

                id=4;
                mmlfits{id}.params.mu = theta{id}(1);
                mmlfits{id}.params.sigma1 = sqrt(theta{id}(2));
                mmlfits{id}.params.sigma2 = sqrt(theta{id}(3));
                mmlfits{id}.params.delta = theta{id}(4);
                mmlfits{id}.codelength = mml_vals(id);
                mmlfits{id}.model_id = id;                  

                % Select models
                [~, m_bic] = min(bic_vals);
                [~, m_mml] = min(mml_vals);

                acc_bic(rep) = (m_bic == gen_model);
                acc_mml(rep) = (m_mml == gen_model);

                kl_bic(rep) = compute_kl(gen_model, fits{m_bic}, params, sample_sizes);
                kl_mml(rep) = compute_kl(gen_model, mmlfits{m_mml}, params, sample_sizes);
            end

            % Store results
            key = sprintf('M%d_d%g_r%g', gen_model, delta, r);
            key = strrep(key, '.', '_');  % Replace decimal points with underscores
            results.(key) = struct( ...
                'accuracy_bic', mean(acc_bic), ...
                'accuracy_mml', mean(acc_mml), ...
                'accuracy_bic_iqr', std(acc_bic), ...
                'accuracy_mml_iqr', std(acc_mml), ...
                'kl_bic_median', mean(kl_bic), ...
                'kl_mml_median', mean(kl_mml), ...
                'kl_bic_iqr', std(kl_bic), ...
                'kl_mml_iqr', std(kl_mml) ...
            );
        end
    end
end

% === Display summary ===
disp('=== Summary ===');
keys = fieldnames(results);
for i = 1:length(keys)
    k = keys{i};
    r = results.(k);
    fprintf('%20s | Acc BIC: %.3f | Acc MML: %.3f | KL BIC: %.3f ± %.3f | KL MML: %.3f ± %.3f\n', ...
        k, r.accuracy_bic, r.accuracy_mml, r.kl_bic_median, r.kl_bic_iqr, r.kl_mml_median, r.kl_mml_iqr);
end

end


function [y1, y2] = simulate_data(model_id, sample_sizes, params)
%SIMULATE_DATA Generate synthetic data from one of four Gaussian models
%   Inputs:
%     model_id      - integer from 1 to 4 indicating the generating model
%     sample_sizes  - [n1, n2] sample sizes for groups 1 and 2
%     params        - struct with fields depending on model:
%                     M1: mu, sigma
%                     M2: mu, sigma1, sigma2
%                     M3: mu, sigma, delta
%                     M4: mu, sigma1, sigma2, delta
%   Outputs:
%     y1, y2        - column vectors of simulated data for groups 1 and 2

n1 = sample_sizes(1);
n2 = sample_sizes(2);

switch model_id
    case 1  % M1: equal means, equal variances
        mu = params.mu;
        sigma = params.sigma;
        y1 = mu + sigma * randn(n1, 1);
        y2 = mu + sigma * randn(n2, 1);

    case 2  % M2: equal means, unequal variances
        mu = params.mu;
        sigma1 = params.sigma1;
        sigma2 = params.sigma2;
        y1 = mu + sigma1 * randn(n1, 1);
        y2 = mu + sigma2 * randn(n2, 1);

    case 3  % M3: mean shift, equal variances
        mu = params.mu;
        sigma = params.sigma;
        delta = params.delta;
        mu1 = mu + (sigma * delta) / 2;
        mu2 = mu - (sigma * delta) / 2;
        y1 = mu1 + sigma * randn(n1, 1);
        y2 = mu2 + sigma * randn(n2, 1);

    case 4  % M4: mean shift, unequal variances
        mu = params.mu;
        sigma1 = params.sigma1;
        sigma2 = params.sigma2;
        delta = params.delta;
        mu1 = mu + (sqrt(sigma1 * sigma2) * delta) / 2;
        mu2 = mu - (sqrt(sigma1 * sigma2) * delta) / 2;
        y1 = mu1 + sigma1 * randn(n1, 1);
        y2 = mu2 + sigma2 * randn(n2, 1);

    otherwise
        error('Invalid model_id: must be 1 to 4');
end
end

function fit = fit_model(model_id, y1, y2)
%FIT_MODEL Estimate parameters and log-likelihood for one of four models
%   Inputs:
%     model_id - integer from 1 to 4
%     y1, y2   - column vectors of observed data
%   Output:
%     fit      - struct with fields:
%                .params: estimated parameters
%                .loglik: log-likelihood at MLE
%                .model_id: identifier

n1 = length(y1);
n2 = length(y2);
N = n1 + n2;

switch model_id
    case 1  % M1: equal mean, equal variance
        mu_hat = mean([y1; y2]);
        sigma_hat = std([y1; y2], 1);  % MLE uses 1/N normalization
        loglik = -N/2 * log(2*pi*sigma_hat^2) ...
                 - sum(([y1; y2] - mu_hat).^2) / (2*sigma_hat^2);
        fit.params = struct('mu', mu_hat, 'sigma', sigma_hat);

    case 2  % M2: equal mean, unequal variances
        mu_hat = mean([y1; y2]);
        sigma1_hat = std(y1, 1);
        sigma2_hat = std(y2, 1);
        loglik = -n1/2 * log(2*pi*sigma1_hat^2) ...
                 - sum((y1 - mu_hat).^2) / (2*sigma1_hat^2) ...
                 -n2/2 * log(2*pi*sigma2_hat^2) ...
                 - sum((y2 - mu_hat).^2) / (2*sigma2_hat^2);
        fit.params = struct('mu', mu_hat, 'sigma1', sigma1_hat, 'sigma2', sigma2_hat);

    case 3  % M3: mean shift, equal variance
        % Initial guesses
        mu0 = mean([y1; y2]);
        sigma0 = std([y1; y2], 1);
        delta0 = (mean(y1) - mean(y2)) / sigma0;

        theta0 = [mu0, log(sigma0), delta0];
        opts = optimset('Display','off');
        [theta_hat, negloglik] = fminsearch(@(theta) negloglik_m3(theta, y1, y2), theta0, opts);

        mu_hat = theta_hat(1);
        sigma_hat = exp(theta_hat(2));
        delta_hat = theta_hat(3);
        fit.params = struct('mu', mu_hat, 'sigma', sigma_hat, 'delta', delta_hat);
        loglik = -negloglik;

    case 4  % M4: mean shift, unequal variances
        mu0 = mean([y1; y2]);
        sigma1_0 = std(y1, 1);
        sigma2_0 = std(y2, 1);
        delta0 = (mean(y1) - mean(y2)) / sqrt(sigma1_0 * sigma2_0);

        theta0 = [mu0, log(sigma1_0), log(sigma2_0), delta0];
        opts = optimset('Display','off');
        [theta_hat, negloglik] = fminsearch(@(theta) negloglik_m4(theta, y1, y2), theta0, opts);

        mu_hat = theta_hat(1);
        sigma1_hat = exp(theta_hat(2));
        sigma2_hat = exp(theta_hat(3));
        delta_hat = theta_hat(4);
        fit.params = struct('mu', mu_hat, 'sigma1', sigma1_hat, 'sigma2', sigma2_hat, 'delta', delta_hat);
        loglik = -negloglik;

    otherwise
        error('Invalid model_id: must be 1 to 4');
end

fit.loglik = loglik;
fit.model_id = model_id;
end

function nll = negloglik_m3(theta, y1, y2)
mu = theta(1);
sigma = exp(theta(2));
delta = theta(3);

mu1 = mu + (sigma * delta)/2;
mu2 = mu - (sigma * delta)/2;

n1 = length(y1);
n2 = length(y2);

nll = n1/2 * log(2*pi*sigma^2) + sum((y1 - mu1).^2) / (2*sigma^2) ...
    + n2/2 * log(2*pi*sigma^2) + sum((y2 - mu2).^2) / (2*sigma^2);
end

function nll = negloglik_m4(theta, y1, y2)
mu = theta(1);
sigma1 = exp(theta(2));
sigma2 = exp(theta(3));
delta = theta(4);

mu1 = mu + (sqrt(sigma1 * sigma2) * delta)/2;
mu2 = mu - (sqrt(sigma1 * sigma2) * delta)/2;

n1 = length(y1);
n2 = length(y2);

nll = n1/2 * log(2*pi*sigma1^2) + sum((y1 - mu1).^2) / (2*sigma1^2) ...
    + n2/2 * log(2*pi*sigma2^2) + sum((y2 - mu2).^2) / (2*sigma2^2);
end

function bic = compute_bic(fit, sample_sizes)
%COMPUTE_BIC Compute BIC for a fitted model
%   Inputs:
%     fit           - struct from fit_model with .loglik and .params
%     sample_sizes  - [n1, n2] sample sizes
%   Output:
%     bic           - scalar BIC value

    N = sum(sample_sizes);

    % Count number of parameters
    param_fields = fieldnames(fit.params);
    k = numel(param_fields);

    % Compute BIC
    bic = -2 * fit.loglik + k * log(N);
end

function kl = compute_kl(true_model, fit, true_params, sample_sizes)
%COMPUTE_KL Compute average KL divergence between true and fitted distributions
%   Inputs:
%     true_model   - integer (1 to 4) indicating the generating model
%     fit          - struct with .params from selected model
%     true_params  - struct with true parameters used to simulate data
%     sample_sizes - [n1, n2] sample sizes
%   Output:
%     kl           - scalar average KL divergence across both groups

n1 = sample_sizes(1);
n2 = sample_sizes(2);
N = n1 + n2;

% === True distributions ===
switch true_model
    case 1
        m1_true = true_params.mu;
        m2_true = true_params.mu;
        s1_true = true_params.sigma;
        s2_true = true_params.sigma;
    case 2
        m1_true = true_params.mu;
        m2_true = true_params.mu;
        s1_true = true_params.sigma1;
        s2_true = true_params.sigma2;
    case 3
        mu = true_params.mu;
        sigma = true_params.sigma;
        delta = true_params.delta;
        m1_true = mu + (sigma * delta)/2;
        m2_true = mu - (sigma * delta)/2;
        s1_true = sigma;
        s2_true = sigma;
    case 4
        mu = true_params.mu;
        sigma1 = true_params.sigma1;
        sigma2 = true_params.sigma2;
        delta = true_params.delta;
        m1_true = mu + (sqrt(sigma1 * sigma2) * delta)/2;
        m2_true = mu - (sqrt(sigma1 * sigma2) * delta)/2;
        s1_true = sigma1;
        s2_true = sigma2;
end

% === Fitted distributions ===
p = fit.params;
switch fit.model_id
    case 1
        m1_fit = p.mu;
        m2_fit = p.mu;
        s1_fit = p.sigma;
        s2_fit = p.sigma;
    case 2
        m1_fit = p.mu;
        m2_fit = p.mu;
        s1_fit = p.sigma1;
        s2_fit = p.sigma2;
    case 3
        mu = p.mu;
        sigma = p.sigma;
        delta = p.delta;
        m1_fit = mu + (sigma * delta)/2;
        m2_fit = mu - (sigma * delta)/2;
        s1_fit = sigma;
        s2_fit = sigma;
    case 4
        mu = p.mu;
        sigma1 = p.sigma1;
        sigma2 = p.sigma2;
        delta = p.delta;
        m1_fit = mu + (sqrt(sigma1 * sigma2) * delta)/2;
        m2_fit = mu - (sqrt(sigma1 * sigma2) * delta)/2;
        s1_fit = sigma1;
        s2_fit = sigma2;
end

% === KL divergence for each group ===
kl1 = computeKL(m1_true, s1_true, m1_fit, s1_fit);
kl2 = computeKL(m2_true, s2_true, m2_fit, s2_fit);

% === Weighted average KL ===
kl = (n1 * kl1 + n2 * kl2) / N;
end
