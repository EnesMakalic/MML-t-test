This is a Matlab project implementing the Minimum Message Length (MML) t-test
from the paper:

Enes Makalic, Daniel F. Schmidt, Minimum Message Length t-test, AJCAI2025: Australasian joint conference on artificial intelligence, 2025.


The code implements a statistical test for discriminating between two groups based on a numerical 
target/outcome variable. The observed data is: y1 ~ Normal [a vector of size n1 x 1],    y2 ~ Normal [a vector of size n2 x 1]
   
We consider the following four candidate models:

   model1: common mean and common s.d.
   
   model2: common mean and different s.d.
   
   model3: different means and common s.d.
   
   model4: different means and different s.d.
   

MML requires prior distributions on all parameters. We use a uniform,
location invariant prior for the grand mean mu. The standard deviations
follow a standard (zero mean, unit scale) half-Cauchy prior distribution. 
The effect size delta follows a standard Cauchy distribution.

The function returns MML estimates of all model parameters for each of
the four models. It also returns the codelengths of each model. The model
with the shortest codelength is deemed optimal. Note that the popular 
Bayesian Information Criterion (BIC) can be seen as an asymptotic approximation 
to the MML codelength used in this function.

The main function implementing the MML t-test is mmlttest.m. An example run is:


y1 = normrnd(0,2,25,1);

y2 = normrnd(5,3,15,1);

[codelengths, theta] = mmlttest(y1, y2, verbose=true);


To reproduce the results in Table 1, one should use the code in run_model_selection_experiment.m.
