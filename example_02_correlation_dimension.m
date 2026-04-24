%% Example 2 - Correlation Dimension
% chrona-scope repository

clc; clear; close all;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Case 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Lorenz system correlation dimension

% Add path to the Lorenz function
addpath('experiments\chaotic_systems\');
addpath("utils\")
addpath("correlation_dimension\")

% Simulation parameters
t_end = 100;                  % total simulation time
N = 10000;                     % number of time steps
X0 = [1,0,20];               % initial condition
method = "Euler";             % integration method
param = [10; 28; 8/3];       % Lorenz parameters [sigma; rho; beta]

% Generate Lorenz attractor time series
[X,t,info] = LorenzAttractor(t_end,N,X0,method,param);

% Embedding variable 2
Xemb = Embedding(X,2,10);

% Correlation dimension
C2_Procaccia = correlationDimension(X);
C2_Krakovska = CorrelationDimension_Krakosvka(X);

% Reverse embedding
Xrev = ReverseEmbedding(Xemb,10,1);



