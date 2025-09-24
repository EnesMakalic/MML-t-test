%MMLTTTEST Minimum Message Length t-test
% Implements the Bayesian MML87 test for discriminating between two groups
% based on a numerical target/outcome variable. The observed data is:
%
%   y1 ~ Normal [n1 x 1]
%   y2 ~ Normal [n2 x 1]
%
% We consider the following four candidate models:
%   model1: common mean and common s.d.         y1 ~ N(mu, sigma^2) and y2 ~ N(mu, sigma^2)
%   model2: common mean and different s.d.      y1 ~ N(mu, sigma1^2) and y2 ~ N(mu, sigma2^2)
%   model3: different means and common s.d.     y1 ~ N(mu + (sigma/2)delta, sigma^2) and y2 ~ N(mu - (sigma/2)delta, sigma^2)
%   model4: different means and different s.d.  y1 ~ N(mu + (sqrt(sigma1 sigma2)/2)delta, sigma1^2) and y2 ~ N(mu - (sqrt(sigma1 sigma2)/2)delta, sigma2^2)
%
% MML87 requires prior distributions on all parameters. We use a uniform,
% location invariant prior for the grand mean mu. The standard deviations
% follow a standard (zero mean, unit scale) half-Cauchy prior distribution. 
% The effect size delta follows a standard Cauchy distribution.
%
% The function returns MML estimates of all model parameters for each of
% the four models. It also returns the codelengths of each model. The model
% with the shortest codelength is deemed optimal. Note that the popular 
% Bayesian Information Criterion (BIC) can be seen as an asymptotic approximation 
% to the MML codelength used in this function.
% 
% Returns:
%   codelengths = [msglen0,msglen1,msglen2,msglen3] 
%   msglen1 - codelength of the null hypothesis: y1 ~ N(mu, sigma^2) and y2 ~ N(mu, sigma^2)
%   msglen2 - codelength of alt. model 1:        y1 ~ N(mu, sigma1^2) and y2 ~ N(mu, sigma2^2)
%   msglen3 - codelength of alt. model 2:        y1 ~ N(mu + (sigma/2)delta, sigma^2) and y2 ~ N(mu - (sigma/2)delta, sigma^2)
%   msglen4 - codelength of alt. model 3:        y1 ~ N(mu + (sqrt(sigma1 sigma2)/2)delta, sigma1^2) and y2 ~ N(mu - (sqrt(sigma1 sigma2)/2)delta, sigma2^2)
% 
%   Parameters:
%   theta = {theta0,theta1,theta2,theta3}
%   theta0 = [mu, sigma^2]
%   theta1 = [mu, sigma1^2, sigma2^2]
%   theta2 = [mu, sigma^2, delta]
%   theta3 = [mu, sigma1^2, sigma2^2, delta]
%
% Example:
% y1 = normrnd(0,2,25,1);
% y2 = normrnd(5,3,15,1);
% [codelengths, theta] = mmlttest(y1, y2, verbose=true);
%
% Cite:
% Minimum message length t-test
% Enes Makalic and Daniel F. Schmidt, AJCAI 2025
%
% Enes Makalic and Daniel Schmidt, 2025-
function [codelengths, theta] = mmlttest(y1, y2, opts)
arguments
        y1 (:,1) double {mustBeNumeric,mustBeReal}
        y2 (:,1) double {mustBeNumeric,mustBeReal}
        opts.minoptions = [] % optional, improves performance when this function is called many times
        opts.verbose = false % set to true if wanting verbose output
end

%% Setup
if(isempty(opts.minoptions))
    opts.minoptions = optimoptions('fminunc','Display','off');
end

n1 = length(y1);
n2 = length(y2);

y = [y1(:); y2(:)];
n = length(y);

theta = cell(1,4);

%% model1: common mean and common s.d.
mu = mean(y); % common mean
% s2 = sum((y - mu).^2) / (n-1); % var estimate if prior is 1/s.d.

% var estimate if prior is half Cauchy
termA = 2 + sum(y.^2) - 2*sum(y)*mu + n*(-1 + mu^2);
termB = 4*n*(sum(y.^2) + mu*(-2*sum(y) + n*mu));
numerator = termA + sqrt(termB + termA^2);
denominator = 2*n;
s2 = numerator / denominator;

negll = n/2*log(2*pi*s2) + 1/2/s2*sum((y - mu).^2);
J     = log(2*n^2)/2 - log(s2);
h = -log(2) + log(pi) + log1p(s2);
msglen0 = negll + J + h + mml_const(2);
theta0 = [mu, s2]; % [mu, sigma1^2]

%% model2: common mean and different s.d.
[theta1, msglen1] = fminunc(@(X) msglen_alt1(y1,y2,X(1),X(2),X(3)), [mean(y),log(var(y1)),log(var(y2))], opts.minoptions);
theta1(2:3) = exp(theta1(2:3));

%% model3: different means and common s.d.
S1 = sum(y1);
S2 = sum(y2);
Z  = sum(y.^2);
mu_ml     = (S1/n1 + S2/n2) / 2;
sigma_ml2 = (Z - (S1^2/n1 + S2^2/n2)) /(n1 + n2);
[theta2, msglen2] = fminunc(@(X) msglen_alt2(y1,y2,X(1),X(2),X(3)), [mu_ml,log(sigma_ml2),0], opts.minoptions);
theta2(2) = exp(theta2(2));

%% model4: different means and different s.d.
[T, msglen3] = fminunc(@(X) msglen_alt3(y1,y2,X(1),X(2),X(3),X(4)), [mean(y),log(var(y1)),log(var(y2)),0], opts.minoptions);
theta3 = [T(1), exp(T(2:end-1)), T(end)];

% Pack results to return
codelengths = [msglen0,msglen1,msglen2,msglen3];
theta{1} = theta0;
theta{2} = theta1;
theta{3} = theta2;
theta{4} = theta3;

%% Display results if required
if(opts.verbose)
    [~,p0,stat0] = ttest2(y1,y2,'vartype','equal');
    [~,p1,stat1] = ttest2(y1,y2,'vartype','unequal');

    models = strings(1,4);
    models(1) = "common mean and common s.d.";
    models(2) = "common mean and different s.d.";
    models(3) = "different means and common s.d.";
    models(4) = "different means and different s.d.";

    fprintf('MML t-test\n');
    fprintf('==========\n');
    fprintf('Sample statistics:\n');
    fprintf('y1 : mean=%+.2f s.d.=%.2f samples=%d\n', mean(y1),std(y1),n1);
    fprintf('y2 : mean=%+.2f s.d.=%.2f samples=%d\n', mean(y2),std(y2),n2);
    fprintf('T-test (equal variance)  : tstat = %.2f p-val=%.2e\n', stat0(1), p0);
    fprintf('T-test (unequal variance): tstat = %.2f p-val=%.2e\n', stat1(1), p1);
    fprintf('\n');
    fprintf('Model 1: y1 ~ N(mu, sigma^2) and y2 ~ N(mu, sigma^2)\n');
    fprintf('\tmu     = %.2f\n', theta0(1));
    fprintf('\tsigma  = %.2f\n', sqrt(theta0(2)));
    fprintf('\tCodelength = %.2f nits\n', msglen0);

    fprintf('Model 2: y1 ~ N(mu, sigma1^2) and y2 ~ N(mu, sigma2^2)\n');
    fprintf('\tmu     = %.2f\n', theta1(1));
    fprintf('\tsigma1 = %.2f\n', sqrt(theta1(2)));
    fprintf('\tsigma2 = %.2f\n', sqrt(theta1(3)));
    fprintf('\tCodelength = %.2f nits\n', msglen1);   

    fprintf('Model 3: y1 ~ N(mu1, sigma^2) and y2 ~ N(mu2, sigma^2)\n');
    s2 = sqrt(theta2(2));
    delta = theta2(3);    
    eff = (s2)/2*delta;    
    mu1 = theta2(1) + eff;
    mu2 = theta2(1) - eff;
    fprintf('\tmu1     = %.2f\n', mu1);
    fprintf('\tmu2     = %.2f\n', mu2);
    fprintf('\tsigma   = %.2f\n', s2);
    fprintf('\tmu1-mu2 = %.2f\n', eff);
    fprintf('\tCodelength = %.2f nits\n', msglen2);     

    fprintf('Model 4: y1 ~ N(mu1, sigma1^2) and y2 ~ N(mu2, sigma2^2)\n');
    s1 = sqrt(theta3(2));
    s2 = sqrt(theta3(3));
    delta = theta3(4);
    eff = sqrt(s1*s2)/2*delta;
    mu1 = theta3(1) + eff;
    mu2 = theta3(1) - eff;
    fprintf('\tmu1     = %.2f\n', mu1);
    fprintf('\tmu2     = %.2f\n', mu2);
    fprintf('\tsigma1  = %.2f\n', s1);
    fprintf('\tsigma2  = %.2f\n', s2);
    fprintf('\tmu1-mu2 = %.2f\n', eff);
    fprintf('\tCodelength = %.2f nits\n', msglen3);   

    fprintf('\n');
    ind = find(codelengths == min(codelengths));
    fprintf('Model with the smallest codelength is:\n');
    fprintf('\tModel %d [%s]\n', ind, models(ind));

    fprintf('\n');
    fprintf('Approximate posterior odds in favour of Models 2-4 over Model 1:\n');
    podds = exp(-(codelengths - codelengths(1)));
    fprintf('\tModel 2: %10.3f\n', podds(2));
    fprintf('\tModel 3: %10.3f\n', podds(3));
    fprintf('\tModel 4: %10.3f\n', podds(4));
end

end

%% Msglen - common mean and different s.d. 
% v1 = s_1^2, v2 = s_2^2
function msglen = msglen_alt1(y1, y2, mu, log_v1, log_v2)

n1 = length(y1);
n2 = length(y2);

v1 = exp(log_v1);
v2 = exp(log_v2);

negll = n1/2*log(2*pi*v1) + 1/2/v1*sum((y1 - mu).^2) + n2/2*log(2*pi*v2) + 1/2/v2*sum((y2 - mu).^2);
J = log(4*n1*n2)/2  + log(n2*v1 + n1*v2)/2 - log_v1 - log_v2;
h = -log(2) + log(pi) + log1p(v1) -log(2) + log(pi) + log1p(v2);
msglen = negll + J + h + mml_const(3);

end

%% Msglen - different means and common s.d.
function msglen = msglen_alt2(y1, y2, mu, log_s2, delta)

n1 = length(y1);
n2 = length(y2);
n  = n1+n2;

s2 = exp(log_s2);
s = sqrt(s2);

% codelength
negll  = n/2*log(2*pi*s2) + 1/2/s2*sum((y1 - mu - s*delta/2).^2) + 1/2/s2*sum((y2 - mu + s*delta/2).^2);
J      = log(2*n1*n2*n)/2 - log(s2);
hdelta = -log(1) + log(pi) + log1p(delta^2);
h = -log(2) + log(pi) + log1p(s2) + hdelta;

msglen = negll + J + h + mml_const(3);

end

%% Msglen - different means and different s.d.
function msglen = msglen_alt3(y1, y2, mu, log_v1, log_v2, delta)

n1 = length(y1);
n2 = length(y2);

v1 = exp(log_v1); sigma1 = sqrt(v1); 
v2 = exp(log_v2); sigma2 = sqrt(v2); 

negll = n1/2*log(2*pi*v1) + 1/2/v1*sum((y1 - mu - sqrt(sigma1*sigma2)/2*delta).^2) + ...
        n2/2*log(2*pi*v2) + 1/2/v2*sum((y2 - mu + sqrt(sigma1*sigma2)/2*delta).^2);

J = log(4)/2 + log(n1) + log(n2) - 3/2*log(sigma1) - 3/2*log(sigma2);

hdelta = -log(1) + log(pi) + log1p(delta^2);
h = -log(2) + log(pi) + log1p(v1) -log(2) + log(pi) + log1p(v2) + hdelta;
msglen = negll + J + h + mml_const(4);

end
