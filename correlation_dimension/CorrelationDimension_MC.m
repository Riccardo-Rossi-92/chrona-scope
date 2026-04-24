function [D2_mean, D2_std, Loss] = CorrelationDimension_MC(X,MC,type,perc)
% CorrelationDimension_MC computes the correlation dimension of a dataset X
% using Monte Carlo sampling with various estimation methods.
%
% [D2_mean,D2_std,Loss] = CorrelationDimension_MC(X,MC,type,perc)
%
% Inputs:
%   X    : data matrix (N x d)
%   MC   : number of Monte Carlo repetitions (default 30)
%   type : method ("Procaccia","Krakosvka","TorVergata")
%   perc : fraction of points to discard in Monte Carlo (default 0.1)
%
% Outputs:
%   D2_mean : mean correlation dimension
%   D2_std  : standard deviation of correlation dimension
%   Loss    : placeholder (currently zero, can be used for fitting error)

%% Input defaults
if nargin < 2 || isempty(MC)
    MC = 30;
end
if nargin < 3 || isempty(type)
    type = "Krakosvka";
end
if nargin < 4 || isempty(perc)
    perc = 0.1;
end

%% Initialize
N = size(X,1);
ind = 1:N;
D2_values = zeros(MC,1); % preallocate
Loss = 0; % placeholder

%% Monte Carlo computation
switch type
    case "Procaccia"
        for mc = 1:MC
            ind_rand = randsample(ind,floor((1-perc)*N));
            D2_values(mc) = correlationDimension(X(ind_rand,:),0,1);
        end

    case "Krakosvka"
        for mc = 1:MC
            ind_rand = randsample(ind,floor((1-perc)*N));
            D2_values(mc) = CorrelationDimension_Krakosvka(X(ind_rand,:));
        end

    case "TorVergata"
        for mc = 1:MC
            ind_rand = randsample(ind,floor((1-perc)*N));
            D2_values(mc) = CorrelationDimension_TorVergata(X(ind_rand,:));
        end
    otherwise
        error('Unknown method type. Use Procaccia, Krakosvka, TorVergata, IDEA, IDEA-2, or TwoNN.')
end

%% Compute statistics
D2_mean = mean(D2_values);
D2_std  = std(D2_values);

end
