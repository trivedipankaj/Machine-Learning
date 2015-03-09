function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%



%take it from exercise 1

h_theta = X * theta;
n = size(theta);
J = 1 / (2 * m) * sum((h_theta - y) .^ 2) + lambda / (2 * m) * sum(theta(2:n) .^ 2);

grad = 1 / m * X' * (h_theta - y);
grad(2:n) = grad(2:n) + lambda / m * theta(2:n);






% =========================================================================

grad = grad(:);

end
