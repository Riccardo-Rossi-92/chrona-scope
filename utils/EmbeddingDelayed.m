function Xemb = EmbeddingDelayed(x, window, delays, variable)
% embedding_delayed generates delayed embeddings from a time series
%
% Inputs:
%   x        : time series (N x d) or (N x 1)
%   window   : embedding window size
%   delays   : vector of delays
%   variable : number of variables (default 1)
%
% Output:
%   Xemb : cell array containing delayed embedded matrices

if nargin < 4 || isempty(variable)
    variable = 1;
end

% Ensure x is column-oriented (time along rows)
if size(x,1) < size(x,2)
    x = x';
end

% Compute base embedding
X = Embedding(x, variable, window);

% Maximum delay
delay_max = max(delays);

% Number of available columns
n_cols = size(X,2);

% Check compatibility
if delay_max >= n_cols
    error('Maximum delay exceeds available embedding length.');
end

% Preallocate cell array
Xemb = cell(length(delays),1);

% Generate delayed embeddings
for j = 1:length(delays)
    d = delays(j);
    
    % Align all embeddings to the same time support
    Xemb{j} = X(:, delay_max - d + 1 : n_cols - d);
end

end
