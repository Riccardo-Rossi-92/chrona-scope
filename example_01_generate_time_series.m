%% Example 1 - Generate Lorenz Time Series
% chrona-scope repository

clc; clear; close all;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Case 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Lorenz system generation

% Add path to the Lorenz function
addpath('experiments\chaotic_systems\');

% Simulation parameters
t_end = 30;                  % total simulation time
N = 5000;                     % number of time steps
X0 = [1,0,20];               % initial condition
method = "Euler";             % integration method
param = [10; 28; 8/3];       % Lorenz parameters [sigma; rho; beta]

% Generate Lorenz attractor time series
[X,t,info] = LorenzAttractor(t_end,N,X0,method,param);

% Plot time series for each variable
figure(1)
clf

subplot(3,2,1)
plot(t,X(:,1),'-b')
grid on; grid minor
xlabel("time [arb.units]"); ylabel("x [arb.units]")
title("Lorenz x(t)")

subplot(3,2,3)
plot(t,X(:,2),'-b')
grid on; grid minor
xlabel("time [arb.units]"); ylabel("y [arb.units]")
title("Lorenz y(t)")

subplot(3,2,5)
plot(t,X(:,3),'-b')
grid on; grid minor
xlabel("time [arb.units]"); ylabel("z [arb.units]")
title("Lorenz z(t)")

subplot(1,2,2)
plot3(X(:,1),X(:,2),X(:,3),'-b')
grid on; grid minor
xlabel("x [arb.units]"); ylabel("y [arb.units]"); zlabel("z [arb.units]")
title("Lorenz attractor - 3D trajectory")
view([45 15])

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Case 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Lorenz Attractor - Chaotic Behaviour Demonstration

% Simulation parameters
t_end = 30;                  % total simulation time
N = 5000;                     % number of time steps
X0 = [1,0,20];               % initial condition
method = "Euler";             % integration method
param = [10; 28; 8/3];       % Lorenz parameters [sigma; rho; beta]

% Generate Lorenz attractor time series
[X,t,info] = LorenzAttractor(t_end,N,X0,method,param);

% Small change in initial condition to show divergence
Y0 = [1.001,0,20];

% Generate new trajectory
[Y,t,info] = LorenzAttractor(t_end,N,Y0,method,param);

% Plot comparison
figure(2)
clf

subplot(3,2,1)
plot(t,X(:,1),'-b'); hold on
plot(t,Y(:,1),'-r'); hold off
grid on; grid minor
xlabel("time [arb.units]"); ylabel("x [arb.units]")
title("Lorenz x(t) - sensitivity to initial condition")

subplot(3,2,3)
plot(t,X(:,2),'-b'); hold on
plot(t,Y(:,2),'-r'); hold off
grid on; grid minor
xlabel("time [arb.units]"); ylabel("y [arb.units]")
title("Lorenz y(t) - sensitivity to initial condition")

subplot(3,2,5)
plot(t,X(:,3),'-b'); hold on
plot(t,Y(:,3),'-r'); hold off
grid on; grid minor
xlabel("time [arb.units]"); ylabel("z [arb.units]")
title("Lorenz z(t) - sensitivity to initial condition")

subplot(1,2,2)
plot3(X(:,1),X(:,2),X(:,3),'-b'); hold on
plot3(Y(:,1),Y(:,2),Y(:,3),'-r'); hold off
grid on; grid minor
xlabel("x [arb.units]"); ylabel("y [arb.units]"); zlabel("z [arb.units]")
title("Lorenz attractor - chaotic divergence 3D view")
view([45 15])
