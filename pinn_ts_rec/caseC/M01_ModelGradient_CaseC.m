function [gradients,Loss,Losses] = M01_ModelGradient_CaseC(parameters,Network,...
                                    dtp,dtm,dfm,dgm,dhm,lambda,sx,sy,sz)


%% Coefficients

gamma = 2;
% 
sigma = 10;
rho = 28;
beta = 8/3;

% sigma = parameters.param.sigma;
% rho =  parameters.param.rho;
% beta = parameters.param.beta;

%% Measurements

X = Network_CaseC(dtm,1,parameters,Network);

dxp = X(1,:);
dyp = X(2,:);
dzp = X(3,:);

dfp = dxp + dyp;
dgp = dxp.*(1+dzp)/50;
% dhp = dyp.*dzp/10;

Z2x = ((dfp-dfm)/sx).^2;
Z2y = ((dgp-dgm)/sy).^2;
% Z2z = ((dhp-dhm)/sz).^2;

% Lx = max(Z2x-1/4,0);
% Ly = max(Z2y-1/4,0);
% Lz = max(Z2z-1/4,0);

Lx = Z2x.^(gamma+1)./(1+Z2x.^gamma);
Ly = Z2y.^(gamma+1)./(1+Z2y.^gamma);
% Lz = Z2z.^(gamma+1)./(1+Z2z.^gamma);

% Loss_meas = mean(Lx+Ly+Lz);
% Loss_meas = mean(Lx+Ly);
% Loss_meas = mean(Lx);
Loss_meas = mean(Lx + Ly);

%% Physics

X = Network_CaseC(dtp,1,parameters,Network);

dxp = X(1,:);
dyp = X(2,:);
dzp = X(3,:);

dxdt = dlgradient(sum(dxp),dtp);
dydt = dlgradient(sum(dyp),dtp);
dzdt = dlgradient(sum(dzp),dtp);

fx = dxdt - sigma.*(dyp-dxp);
fy = dydt - dxp.*(rho-dzp) + dyp; 
fz = dzdt - dxp.*dyp + beta*dzp;

Loss_physics = mean(fx.^2./sx.^2 + fy.^2./sy.^2 + fz.^2./sz.^2);

%% Loss total

Losses = [Loss_meas Loss_physics];
Loss = Loss_meas + lambda.*Loss_physics;

gradients = dlgradient(Loss,parameters);

end