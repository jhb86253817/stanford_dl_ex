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
  %Version1--------------------------------------------------------------------------------
  %1800s, 92.1%
  %y_hat = exp(theta' * X);
  %y_hat = [y_hat; ones(1,m)];
  %I = sub2ind(size(y_hat), y, 1:size(y_hat,2));
  %y_hat_single = y_hat(I);
  %y_hat_sum = sum(y_hat);
  %y_hat_p = y_hat_single ./ y_hat_sum;
  %y_hat_pp = bsxfun(@rdivide, y_hat, y_hat_sum);
  %f = -sum(log(y_hat_p));

  %g = [g, zeros(n,1)];
  %for i = 1:m,
  %  g(:,y(i)) = g(:,y(i)) - X(:,i);
  %end;

  %for i = 1:num_classes,
  %  g(:,i) = g(:,i) + sum(bsxfun(@times, X, y_hat_pp(i,:)), 2);
  %end;

  %g = g(:,1:end-1);
  %Version1--------------------------------------------------------------------------------

  %Version2--------------------------------------------------------------------------------
  %855s, 92.2%
  theta = [theta, zeros(n,1)];
  ground_truth = full(sparse(y, 1:m, 1));
  y_hat = exp(theta' * X);
  y_hat_sum = sum(y_hat);
  y_hat_norm = bsxfun(@rdivide, y_hat, y_hat_sum);
  f = - ground_truth(:)' * log(y_hat_norm(:));

  g = -(ground_truth-y_hat_norm) * X';
  g = g'(:, 1:end-1);
  %Version2--------------------------------------------------------------------------------

  
  g=g(:); % make gradient a vector for minFunc

