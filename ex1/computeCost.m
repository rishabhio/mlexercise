function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
% number of training examples

% You need to return the following variables correctly 
%J = 0;


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


%step-1 compute hypothesis h(theta) for every x and store it in the hypothesis H array

%step-2 add corresponding elements of H and y and create a difference array Z 

%step-3 square all elements of Z and add them together

%step-4 multiply 1/m with the sum obtained in step 3

%step-5 set J = result of step 4
h = ones(1,length(y));
for i = 1:length(y)
	x = ones(1,2);
	x(1,1) = X(i,1);
	x(1,2) = X(i,2);
	h(i) = x * theta;
end
diff = ones(1,length(y));
for i = 1:length(y)
	temp = h(i) - y(i);
	diff(i) = temp * temp;
end
diff;
summation = 0;
for i = 1:length(y)
	summation = summation + diff(i);
end
J = summation / (2 * length(y)); 
% =========================================================================

end
