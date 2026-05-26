%% Physics-Informed Neural Network for Time Series Reconstruction

% Reference
% Title: Physics-Informed Neural Networks for Time-Series Reconstruction 
% and Modelling with Sparse and Indirect Measurements
% Authors: R. Rossi (r.rossi@ing.uniroma2.it), M.Gelfusa, T. Craciunescu
% N. Rutigliano, P. Gaudio, A. Murari, JET Contributors, WPTE Team
% Journal: submitted to Nature Communications

% Case C - Time-series reconstruction from indirect measurements

% Description:
%   Reconstruction of the full state of the Lorenz system (x, y, z) from
%   sparse and noisy indirect measurements. None of the state variables is
%   directly observable. Instead, two nonlinear combinations of the state
%   are measured:
%
%       f(t) = x(t) + y(t)
%       g(t) = x(t) * (1 + z(t)) / 50
%
%   The governing equations and all model parameters (sigma, rho, beta)
%   are assumed to be known. The data-consistency loss is computed on the
%   predicted observables f_p and g_p, obtained by applying the measurement
%   model to the network outputs. The physics residual loss enforces the
%   full Lorenz dynamics at the collocation points (Sobol sequence).
%   To avoid degenerate zero solutions during early training, the physics
%   weight is gradually ramped up via a warm-up schedule.

%% 

clear; clc;

%% generate system

[X,t,info] = LorenzAttractor(10,1e4,[1 1 18]);

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

Lorenz.t = t; 
Lorenz.x = X(:,1);
Lorenz.y = X(:,2);
Lorenz.z = X(:,3);

%% Configuration

Measurements.N = 30;
Measurements.Noise = 0.1;

sx = max(2*Measurements.Noise,0.1);
sy = max(2*Measurements.Noise,0.1);
sz = max(2*Measurements.Noise,0.1);

%% Prepare Measurements Data

% Sample Data

Measurements.t = linspace(min(Lorenz.t),max(Lorenz.t),Measurements.N);

Measurements.x = interp1(Lorenz.t,Lorenz.x,Measurements.t);
Measurements.y = interp1(Lorenz.t,Lorenz.y,Measurements.t);
Measurements.z = interp1(Lorenz.t,Lorenz.z,Measurements.t);

Measurements.f = Measurements.x + Measurements.y;
Measurements.g = Measurements.x.*(1 + Measurements.z)/50;
Measurements.h = Measurements.y.*Measurements.z/1e1;

% Add Noise

Measurements.f = normrnd(Measurements.f,Measurements.Noise);
Measurements.g = normrnd(Measurements.g,Measurements.Noise);
Measurements.h = normrnd(Measurements.h,Measurements.Noise);

% dlarray
dtm = dlarray(Measurements.t,'CB');
dfm = dlarray(Measurements.f,'CB');
dgm = dlarray(Measurements.g,'CB');
dhm = dlarray(Measurements.h,'CB');

%% Physics Grid

PINN.MaximumEpochs = 1e4;
PINN.MiniBatch = 1000;
PINN.IterPerEpoch = 200;
PINN.NumberOfCollocations = PINN.MiniBatch*PINN.IterPerEpoch;

dtp = sobolset(1);
dtp = dtp(1:PINN.NumberOfCollocations)'.*(max(Lorenz.t)-min(Lorenz.t))+min(Lorenz.t);

dtp = dlarray(dtp,'CB');

%% Plot data

dt_plot = dlarray(Lorenz.t,'CB');

%% Neural Network Initialisation

addpath("Network\")

Network.Layer = [20 20 20 20 20 20 20 20];
Network.time_window = [min(Lorenz.t) max(Lorenz.t)];

Network.ScaleX = std(Measurements.x);
Network.ScaleY = std(Measurements.y);
Network.ScaleZ = std(Measurements.z);

[X,parameters] = Network_CaseC(dtp(:,1),0,[],Network);
X = Network_CaseC(dtp(1,1:1000),1,parameters,Network);


%% Model Gradient

addpath("ModelGradients\")
accfun = dlaccelerate(@M01_ModelGradient_CaseC);

%% Training Options and Initialisation

PINN.lambda = 1e-2;
PINN.LearningRate0 = 2e-3; 
PINN.DecayRate0 = 1e-4;

iteration = 0;

averageGrad = [];
averageSqGrad = [];

%% Training

figure(1)
clf

Loss_recorded = zeros(1,PINN.IterPerEpoch);

for epoch = 1 : PINN.MaximumEpochs

    for i = 1 : PINN.IterPerEpoch

        % iteration update 
        iteration = iteration + 1;

        % First Iteration with No Physics to Avoid 0 solutions
        lambda = PINN.lambda.*iteration./(10*PINN.IterPerEpoch+iteration);

        % indices
        ind = ((i-1)*PINN.MiniBatch+1):i*PINN.MiniBatch;

        % Model Gradient
        [gradients,Loss,Losses] = dlfeval(accfun,parameters,Network,...
                                    dtp(1,ind),dtm,dfm,dgm,dhm,PINN.lambda,...
                                    sx,sy,sz);
        % Learning Rate Update
        LearningRate = PINN.LearningRate0./(1+PINN.DecayRate0*iteration);

        % Training Update
        [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
            averageSqGrad,iteration,LearningRate);

        % Record loss
        Loss_recorded(i) = double(extractdata(gather(Loss)));

    end


    %%

    X = Network_CaseC(dt_plot,1,parameters,Network);
    dxp = X(1,:);
    dyp = X(2,:);
    dzp = X(3,:);

    figure(1)
    subplot(2,3,1)
    plot(epoch,Loss,'.k','markersize',16)
    hold on
    plot(epoch,Losses(1),'.b','markersize',16)
    plot(epoch,PINN.lambda.*Losses(2),'.r','markersize',16)
    hold on
    grid on
    grid minor
    xlabel("epoch")
    ylabel("Loss")
    set(gca,'yscale','log')
    legend("Total","Measurements","Physics")

    subplot(2,3,2)
    hold off
    plot(Lorenz.x,Lorenz.y,'b')
    hold on
    plot(dxp,dyp,'r',"LineWidth",1.2)
    grid on
    grid minor
    xlabel("x")
    ylabel("y")

    subplot(2,3,3)
    hold off
    plot(Lorenz.x,Lorenz.z,'b')
    hold on
    plot(dxp,dzp,'r',"LineWidth",1.2)
    grid on
    grid minor
    xlabel("x")
    ylabel("y")

    subplot(2,3,4)
    hold off
    plot(Lorenz.t,Lorenz.x,'b')
    hold on
    plot(dt_plot,dxp,'r',"LineWidth",1.2)
    grid on
    grid minor
    xlabel("time")
    ylabel("Data")

    subplot(2,3,5)
    hold off
    plot(Lorenz.t,Lorenz.y,'b')
    hold on
    % plot(dtm,dym,'.k','MarkerSize',16)
    plot(dt_plot,dyp,'r','LineWidth',1.2)
    grid on
    grid minor
    xlabel("time")
    ylabel("Data")

    subplot(2,3,6)
    hold off
    plot(Measurements.t,Measurements.f,'.b')
    hold on
    plot(dt_plot,dxp+dyp,'r','LineWidth',1.2)
    grid on
    grid minor
    xlabel("time")
    ylabel("Data")

    drawnow

end
