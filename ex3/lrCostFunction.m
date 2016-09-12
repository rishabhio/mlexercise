function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% STEP - 1 Count cost function theta

%h = ones(1,length(y));         % initialize array of ones to be used later as a container
%input1 = X * theta;            % X * theta
%H = sigmoid(input1);           % calculate h(theta)x 
%summation1 = 0;                % initialize a variable to hold summation
%lh = log(H);                   % in case of confusion refer to solutions of assignment 2
%temploh = 1 - H;
%loh = log(temploh);
%minusy = -y;
%oneminusy = 1 - y;
%matrix1 = minusy .* lh;
%matrix2 = loh .* oneminusy;
%seedmatrix = matrix1 - matrix2;
%temp = sum(seedmatrix);
%summation = sum(temp);
%J = summation/m;
%add1 = summation/m;
%regularize = theta(:,2:end) .* theta(:,2:end);
%regusum = sum(regularize);
%sumagain = sum(regusum);
%add2 = sumagain * (lambda/(2*m));

%J = add1 + add2;




% =============================================================

%g = ones(1,length(theta));
%temp1 = H - y;
%temp2 = X' * temp1;
%tempsum = sum(temp2);


%grad = grad(:);
%regugradient = theta .* (lambda/m);
%grad = tempsum + regugradient;

h = sigmoid(X*theta);
theta_reg = [0;theta(2:end, :);];
J = (1/m)*(-y'* log(h) - (1 - y)'*log(1-h))+(lambda/(2*m))*theta_reg'*theta_reg;
grad = (1/m)*(X'*(h-y)+lambda*theta_reg);




end
