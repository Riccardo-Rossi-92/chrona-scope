function Xemb = Embedding(X, variable, window)
% Embedding generates a sliding window embedding for selected variables
%
% Inputs:
%   X        : data matrix (N x d)
%   variable : indices of columns to embed
%   window   : embedding window size
%
% Output:
%   Xemb : matrix where each column is a sliding window from the selected variables

Nvar = numel(variable);       % number of variables
N = size(X,1);                % number of time points
L = N - window;           % number of valid sliding windows

% Preallocate
Xemb = zeros(Nvar*window, L+1);

for i = 1:Nvar
    % Select the i-th variable column
    x = X(:,variable(i));
    
    % Create sliding windows: each column is a window of size 'window'
    Xt = buffer(x, window, window-1, 'nodelay'); 
      
    % Store in the preallocated Xemb
    Xemb((i-1)*window+1:i*window, :) = Xt;
end

end
