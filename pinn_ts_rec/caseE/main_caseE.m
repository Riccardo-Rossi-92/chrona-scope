%% Physics-Informed Neural Network for Time Series Reconstruction

% Reference
% Title: Physics-Informed Neural Networks for Time-Series Reconstruction 
% and Modelling with Sparse and Indirect Measurements
% Authors: R. Rossi (r.rossi@ing.uniroma2.it), M.Gelfusa, T. Craciunescu
% N. Rutigliano, P. Gaudio, A. Murari, JET Contributors, WPTE Team
% Journal: submitted to Nature Communications

matlab%% Physics-Informed Neural Network for Time Series Reconstruction
%
% Reference
% Title:   Physics-Informed Neural Networks for Time-Series Reconstruction
%          and Modelling with Sparse and Indirect Measurements
% Authors: R. Rossi, M. Gelfusa, T. Craciunescu, N. Rutigliano, P. Gaudio,
%          A. Murari, on behalf of JET contributors and EUROfusion
%          Tokamak Exploitation Team
% Journal: [journal name]
% DOI:     [DOI]
%
% Case E - Attractor disentanglement, reconstruction and modelling
%          from mixed dynamics
%
% Description:
%   Simultaneous disentanglement and reconstruction of two independent
%   dynamical systems whose states are observed only through their
%   superposition. The two systems are the Lorenz attractor (3D chaotic)
%   and the Lotka-Volterra system (2D oscillatory). The available
%   observables are the following linear mixtures:
%
%       x_obs(t) = x_L(t) + x_LV(t)
%       y_obs(t) = y_L(t) + y_LV(t)
%       z_obs(t) = z_L(t)
%
%   The parameters rho and beta of the Lorenz system and c and d of the
%   Lotka-Volterra system are unknown and identified simultaneously during
%   training. The PINN predicts the five hidden state variables
%   (x_L, y_L, z_L, x_LV, y_LV) and the four missing parameters as
%   trainable scalars. The measurement loss is computed on the predicted
%   mixtures, while separate physics residual losses enforce the Lorenz
%   and Lotka-Volterra dynamics at the collocation points (Sobol sequence).
%   The incompatible phase-space structures of the two systems ensure
%   identifiability of the decomposition.
%   An adaptive weighting strategy is employed to balance data fidelity
%   and physics regularisation throughout training, adjusting lambda based
%   on the current measurement loss relative to a target MSE threshold.

%%

clear; clc;

%% generate system

[Lorenz,LV] = generate_hybrid_system();

Measurements.N = 100;
Measurements.Noise = 0.5;

sx = max(Measurements.Noise,0.1);
sy = max(Measurements.Noise,0.1);
sz = max(Measurements.Noise,0.1);

%% Configuration

% Data configuration
Config.t_range = [0 10];
Config.t_range(1) = max(min(Lorenz.t),Config.t_range(1));
Config.t_range(2) = min(max(Lorenz.t),Config.t_range(2));

% Weighiting scheme
PINN.lambda_range = [1e-7 inf];
PINN.lambda = 1e-7;
PINN.lambda_adaptive = 1;
PINN.MSE_target = 1.5;

%% Plot data

Plot.rate = 1;

Plot.t = linspace(Config.t_range(1),Config.t_range(2),1000);

Plot.L_x = interp1(Lorenz.t,Lorenz.x,Plot.t);
Plot.L_y = interp1(Lorenz.t,Lorenz.y,Plot.t);
Plot.L_z = interp1(Lorenz.t,Lorenz.z,Plot.t);

Plot.LV_x = interp1(Lorenz.t,LV.x,Plot.t);
Plot.LV_y = interp1(Lorenz.t,LV.y,Plot.t);

%% Prepare Measurements Data

% Sample Data

Measurements.t = linspace(Config.t_range(1),...
    Config.t_range(2),Measurements.N);

Measurements.x = interp1(Lorenz.t,Lorenz.x+LV.x,Measurements.t);
Measurements.y = interp1(Lorenz.t,Lorenz.y+LV.y,Measurements.t);
Measurements.z = interp1(Lorenz.t,Lorenz.z,Measurements.t);

% Add Noise

Measurements.x = normrnd(Measurements.x,Measurements.Noise);
Measurements.y = normrnd(Measurements.y,Measurements.Noise);
Measurements.z = normrnd(Measurements.z,Measurements.Noise);

% dlarray
dtm = dlarray(Measurements.t,'CB');
dxm = dlarray(Measurements.x,'CB');
dym = dlarray(Measurements.y,'CB');
dzm = dlarray(Measurements.z,'CB');

%% Physics Grid

PINN.MaximumEpochs = 1e4;
PINN.MiniBatch = 1000;
PINN.IterPerEpoch = 200;
PINN.NumberOfCollocations = PINN.MiniBatch*PINN.IterPerEpoch;

dtp = sobolset(1);
dtp = dtp(1:PINN.NumberOfCollocations)'.*(Config.t_range(2)-Config.t_range(1)) + ...
    Config.t_range(1);

dtp = dlarray(dtp,'CB');

%% Plot data

dt_plot = dlarray(Plot.t,'CB');

%% Neural Network Initialisation

addpath("Network\")

Network.Layer = [20 20 20 20 20 20 20 20];
Network.time_window = Config.t_range;

Network.ScaleX = max(Measurements.x)-min(Measurements.x);
Network.ScaleY = max(Measurements.y)-min(Measurements.y);
Network.ScaleZ = max(Measurements.z)-min(Measurements.z);

[X,parameters] = Network_CaseE(dtp(:,1),0,[],Network);
X =  Network_CaseE(dtp(1,1:1000),1,parameters,Network);


%% Model Gradient

addpath("ModelGradients\")
accfun = dlaccelerate(@M01_ModelGradient_CaseE);

%% Training Options and Initialisation

PINN.LearningRate0 = 1e-3;
PINN.DecayRate0 = 0;

iteration = 0;

averageGrad = [];
averageSqGrad = [];

%% Training

lambda = PINN.lambda;

figure(1)
clf

Loss_recorded = zeros(1,PINN.IterPerEpoch);

for epoch = 1 : PINN.MaximumEpochs

    for i = 1 : PINN.IterPerEpoch

        % iteration update
        iteration = iteration + 1;

        % indices
        ind = ((i-1)*PINN.MiniBatch+1):i*PINN.MiniBatch;

        % Model Gradient
        [gradients,Loss,Losses] = dlfeval(accfun,parameters,Network,...
            dtp(1,ind),dtm,dxm,dym,dzm,lambda,...
            sx,sy,sz);
        % Learning Rate Update
        LearningRate = PINN.LearningRate0./(1+PINN.DecayRate0*iteration);

        % Training Update
        [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
            averageSqGrad,iteration,LearningRate);

        % Record loss
        Loss_recorded(i) = double(extractdata(gather(Loss)));

    end


    %% plot

    if (epoch/Plot.rate - floor(epoch/Plot.rate)) == 0

        X = Network_CaseE(dt_plot,1,parameters,Network);
        L_xp = X(1,:);
        L_yp = X(2,:);
        L_zp = X(3,:);
        LV_xp = X(4,:);
        LV_yp = X(5,:);

        figure(1)
        subplot(2,3,1)
        plot(epoch,Loss,'.k','markersize',16)
        hold on
        plot(epoch,Losses(1),'.b','markersize',16)
        plot(epoch,lambda.*Losses(2)./(1+lambda),'.r','markersize',16)
        hold on
        grid on
        grid minor
        xlabel("epoch")
        ylabel("Loss")
        set(gca,'yscale','log')
        legend("Total","Measurements","Physics")

        subplot(2,3,2)
        hold off
        plot(Plot.t,Plot.L_x+Plot.LV_x,'-b')
        hold on
        plot(dt_plot,L_xp+LV_xp,'.b','MarkerSize',12)
        plot(Measurements.t,Measurements.x,'.k','MarkerSize',16)
        grid on
        grid minor
        xlabel("t")
        ylabel("x")

        subplot(2,3,3)
        hold off
        plot(Plot.t,Plot.L_y+Plot.LV_y,'-b')
        hold on
        plot(dt_plot,L_yp+LV_yp,'.b','MarkerSize',12)
        plot(Measurements.t,Measurements.y,'.k','MarkerSize',16)
        grid on
        grid minor
        xlabel("t")
        ylabel("x")

        subplot(2,3,4)
        hold off
        plot(Plot.t,Plot.L_x,'-b')
        hold on
        plot(Plot.t,Plot.LV_x,'-r')
        plot(dt_plot,L_xp,'.b','MarkerSize',12)
        plot(dt_plot,LV_xp,'.r','MarkerSize',12)
        grid on
        grid minor
        xlabel("t")
        ylabel("x")

        subplot(2,3,5)
        hold off
        plot(Plot.t,Plot.L_y,'-b')
        hold on
        plot(Plot.t,Plot.LV_y,'-r')
        plot(dt_plot,L_yp,'.b','MarkerSize',12)
        plot(dt_plot,LV_yp,'.r','MarkerSize',12)
        grid on
        grid minor
        xlabel("t")
        ylabel("y")

        subplot(2,3,6)
        hold off
        plot(Plot.t,Plot.L_z,'-b')
        hold on
        plot(dt_plot,L_zp,'.b','MarkerSize',12)
        grid on
        grid minor
        xlabel("t")
        ylabel("z")

        drawnow
    end
    % disp("epoch: " + epoch)
    % disp("beta: " + double(extractdata(gather(parameters.param.beta))))
    % % disp("rho: " + double(extractdata(gather(parameters.param.rho))))
    % disp("sigma: " + double(extractdata(gather(parameters.param.sigma))))

    %% update lambda

    if PINN.lambda_adaptive == 1

        MSE = double(extractdata(gather(Losses(1))));
        lambda = lambda.*(1 - 0.1*tanh(MSE./PINN.MSE_target-1));

        lambda = max(lambda,PINN.lambda_range(1));
        lambda = min(lambda,PINN.lambda_range(2));

        disp("lambda :")
        disp(lambda)

    end

end
