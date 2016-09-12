function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha



% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);


%for iter = 1:num_iters
for iter = 1:num_iters
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.   

% calculate common things


%calculate for theta(1)

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
	diff(i) = temp;
end
addables = ones(1,length(y));
for i = 1:length(y)
	temp = diff(i) * X(i,1);
	addables(i) = temp;
end
summation1 = 0;
for i = 1:length(y)
	summation1 = summation1 + addables(i);
end

theta(1) = theta(1) - ( alpha * (1/m) * summation1 );



%calculate for theta(2)
addables = ones(1,length(y));
for i = 1:length(y)
	temp = diff(i) * X(i,2);
	addables(i) = temp;
end
summation2 = 0;
for i = 1:length(y)
	summation2 = summation2 + addables(i);
end
theta(2) = theta(2) - ( alpha * (1/m) * summation2 );

%cost = computeCost(X,y,theta)




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
end

%end
