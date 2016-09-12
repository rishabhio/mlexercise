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
h = ones(1,length(y));      % initializing an array of ones
input = X * theta;           % X * theta
H = sigmoid(input);         % calculate h(theta)x 

summation = 0;
for i = 1:length(y)
	lh = log(H(i));
	loh = log(1-H(i));
	temp = -y(i)*lh - loh*(1-y(i));
	summation = summation + temp;
end


sumtheta = 0;
for i = 2:length(theta)
	sumtheta = sumtheta + (theta(i)*theta(i));
end
J = summation/m + (lambda/(2*m))*(sumtheta);


g = ones(1,length(theta));
for j = 1:length(theta)
	sum = 0;
	for i = 1:length(y)
		temp1 = H(i) - y(i); 
		temp2 = X(i,j) * temp1;
		sum = sum + temp2;
	end
	if j==1
	g(j) = sum/m;
	else
	g(j) = sum/m + (lambda/m)*theta(j);
	end
end

grad = g;



% =============================================================

end
