function X = ReverseEmbedding(Xembedded, embedding_window, variable)
% ReverseEmbedding reconstructs the original time series from an embedded matrix
%
% Inputs:
%   Xembedded       : embedded matrix (window*variable x N_windows)
%   embedding_window: size of the embedding window
%   variable        : number of variables (default 1)
%
% Output:
%   X : reconstructed time series (N x variable)

if nargin < 3 || isempty(variable)
    variable = 1;
end

% Number of sliding windows
N_windows = size(Xembedded, 2);

% Total length of the original signal
N = N_windows + embedding_window - 1;

% Preallocate output
X = zeros(N, variable);

for var = 1:variable
    % Extract the block for this variable
    block = Xembedded((var-1)*embedding_window+1 : var*embedding_window, :);

    % Accumulate sums and counts for averaging
    sum_vals = zeros(N,1);
    count_vals = zeros(N,1);

    for k = 1:embedding_window
        idx = k : k+N_windows-1;    % indices in original signal
        sum_vals(idx) = sum_vals(idx) + block(k,:)';
        count_vals(idx) = count_vals(idx) + 1;
    end

    % Compute the mean for overlapping windows
    X(:,var) = sum_vals ./ count_vals;
end

end
