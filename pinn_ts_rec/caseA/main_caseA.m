%% Physics-Informed Neural Network for Time Series Reconstruction
%
% Reference
% Title: 
% Authors: 
% Journal: 
% 
% Case A 
% Description

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

Measurements.N = 60;
Measurements.Noise = 0.5;

sx = max(2*Measurements.Noise,0.1);
sy = max(2*Measurements.Noise,0.1);
sz = max(2*Measurements.Noise,0.1);

%% Prepare Measurements Data

% Sample Data

Measurements.t = linspace(min(Lorenz.t),max(Lorenz.t),Measurements.N);

Measurements.x = interp1(Lorenz.t,Lorenz.x,Measurements.t);
Measurements.y = interp1(Lorenz.t,Lorenz.y,Measurements.t);
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
PINN.MiniBatch = 5000;
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

Network.ScaleX = max(Measurements.x);
Network.ScaleY = max(Measurements.y);
Network.ScaleZ = max(Measurements.z);

[X,parameters] = Network_CaseA(dtp(:,1),0,[],Network);
X = Network_CaseA(dtp(1,1:1000),1,parameters,Network);


%% Model Gradient

addpath("ModelGradients\")
accfun = dlaccelerate(@M01_ModelGradient_Lorenz_CaseA);

%% Training Options and Initialisation

PINN.lambda = 1e-4;
PINN.LearningRate0 = 1e-3; 
PINN.DecayRate0 = 1e-5;

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

        % indices
        ind = ((i-1)*PINN.MiniBatch+1):i*PINN.MiniBatch;

        % Model Gradient
        [gradients,Loss,Losses] = dlfeval(accfun,parameters,Network,...
                                    dtp(1,ind),dtm,dxm,dym,dzm,PINN.lambda,...
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

    X = Network_CaseA(dt_plot,1,parameters,Network);
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
    plot(dxm,dym,'.k','MarkerSize',16)
    plot(dxp,dyp,'r',"LineWidth",1.2)
    grid on
    grid minor
    xlabel("x")
    ylabel("y")

    subplot(2,3,3)
    hold off
    plot(Lorenz.x,Lorenz.z,'b')
    hold on
    plot(dxm,dzm,'.k','MarkerSize',16)
    plot(dxp,dzp,'r',"LineWidth",1.2)
    grid on
    grid minor
    xlabel("x")
    ylabel("y")

    subplot(2,3,4)
    hold off
    plot(Lorenz.t,Lorenz.x,'b')
    hold on
    plot(dtm,dxm,'.k','MarkerSize',16)
    plot(dt_plot,dxp,'r',"LineWidth",1.2)
    grid on
    grid minor
    xlabel("time")
    ylabel("Data")

    subplot(2,3,5)
    hold off
    plot(Lorenz.t,Lorenz.y,'b')
    hold on
    plot(dtm,dym,'.k','MarkerSize',16)
    plot(dt_plot,dyp,'r','LineWidth',1.2)
    grid on
    grid minor
    xlabel("time")
    ylabel("Data")

    subplot(2,3,6)
    hold off
    plot(Lorenz.t,Lorenz.z,'b')
    hold on
    plot(dtm,dzm,'.k','MarkerSize',16)
    plot(dt_plot,dzp,'r','LineWidth',1.2)
    grid on
    grid minor
    xlabel("time")
    ylabel("Data")

    drawnow

end
