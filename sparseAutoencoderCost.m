function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 

cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));
%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

 

% n is #of features 64, m is # of samples 10000
[n, m] = size(data);

%use forward propagation to get values of each node of the net work
z2 = W1*data+repmat(b1,1,m); %replicate vector b to m column
a2 = sigmoid(z2);  
z3 = W2*a2+repmat(b2,1,m);  
a3 = sigmoid(z3);  

%calculate cost function
square_err = (0.5/m)*sum(sum((a3-data).^2)); %calculate the square error

% calculate weight penalty
Jweight = (1/2) * (sum(sum(W1.^2))+ sum(sum(W2.^2)));

% calculate sparse penalty
rho = (1/m) .* sum(a2,2);
Jsparse= sum(sparsityParam.*log(sparsityParam./rho)+...
(1-sparsityParam).*log((1-sparsityParam)./(1-rho)));

%calculate the cost function
cost = square_err + lambda*Jweight + beta*Jsparse;

% back propagation algorithm
grad_z3=a3.*(1-a3);  
delta_3=-(data-a3).*grad_z3;  
grad_z2=a2.*(1-a2);  
delta_2=(W2'*delta_3+repmat(beta*(-sparsityParam./rho+(1-sparsityParam)./(1-rho)),1,m)).*grad_z2;  
Delta_W2=delta_3*a2';  
Delta_b2=sum(delta_3,2);  
Delta_W1=delta_2*data';  
Delta_b1=sum(delta_2,2);  
% W1grad is equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}]
W1grad=1/m*Delta_W1+lambda*W1;  
% same for W2grad
W2grad=1/m*Delta_W2+lambda*W2;  
b1grad=1/m*Delta_b1;  
b2grad=1/m*Delta_b2;










%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

