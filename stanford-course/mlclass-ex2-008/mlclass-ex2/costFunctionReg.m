function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
sig = X *theta;
hx = sigmoid(sig);
f = log(hx)' *y + log(1-hx)' * (1-y);
J1 = -1*f/m;

grad1 = X'*(hx-y)/m;

J =J1 + lambda *((theta'*theta) - (theta(1)*theta(1)))/(2*m);
grad = grad1 + (lambda/m)*theta;
grad(1) =grad1(1);
% =============================================================

end
