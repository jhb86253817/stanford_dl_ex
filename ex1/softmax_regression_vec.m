function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%
  %performance on my laptop: 855s, 92.2%
  %add the missing column for digit 10
  theta = [theta, zeros(n,1)];
  %denote the correct class for each training example
  ground_truth = full(sparse(y, 1:m, 1));
  %10*60000 matrix
  y_hat = exp(theta' * X);
  y_hat_sum = sum(y_hat);
  y_hat_norm = bsxfun(@rdivide, y_hat, y_hat_sum);
  %compute cost function
  f = - ground_truth(:)' * log(y_hat_norm(:));

  %compute gradient
  g = -(ground_truth-y_hat_norm) * X';
  %remove the added column for digit 10
  g = g'(:, 1:end-1);

  
  g=g(:); % make gradient a vector for minFunc

