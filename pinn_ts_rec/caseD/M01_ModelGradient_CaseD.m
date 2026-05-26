function [gradients,Loss,Losses] = M01_ModelGradient_CaseD(parameters,Network,...
                                    dtp,dtm,dfm,dgm,dhm,lambda,sx,sy,sz)


%% Coefficients

gamma = 2;
% 
% sigma = 10;
% rho = 28;
% beta = 8/3;

sigma = parameters.param.sigma;
rho =  parameters.param.rho;
beta = parameters.param.beta;
alpha1 = parameters.param.alpha1;
alpha2 = parameters.param.alpha2;
alpha3 = parameters.param.alpha3;

%% Measurements

X = Network_CaseD(dtm,1,parameters,Network);

dxp = X(1,:);
dyp = X(2,:);
dzp = X(3,:);

dfp = dxp + dyp;
dgp = dxp.*(1+dzp.^2)/500;
dhp = dyp.*dzp/10;

Z2x = ((dfp-dfm)/sx).^2;
Z2y = ((dgp-dgm)/sy).^2;
Z2z = ((dhp-dhm)/sz).^2;

% Lx = max(Z2x-1/4,0);
% Ly = max(Z2y-1/4,0);
% Lz = max(Z2z-1/4,0);

Lx = Z2x.^(gamma+1)./(1+Z2x.^gamma);
Ly = Z2y.^(gamma+1)./(1+Z2y.^gamma);
Lz = Z2z.^(gamma+1)./(1+Z2z.^gamma);

Loss_meas = mean(Lx+Ly+Lz);
% Loss_meas = mean(Lx+Ly);
% Loss_meas = mean(Lx);

%% Physics

X = Network_CaseD(dtp,1,parameters,Network);

dxp = X(1,:);
dyp = X(2,:);
dzp = X(3,:);

dxdt = dlgradient(sum(dxp),dtp);
dydt = dlgradient(sum(dyp),dtp);
dzdt = dlgradient(sum(dzp),dtp);

fx = dxdt - sigma.*(dyp-dxp) + alpha1.*dzp;
fy = dydt - dxp.*(rho-dzp) + dyp + alpha2.*dyp; 
fz = dzdt - dxp.*dyp + beta*dzp + alpha3.*dxp;

Loss_physics = mean(fx.^2./sx.^2 + fy.^2./sy.^2 + fz.^2./sz.^2);

%% Loss total

Losses = [Loss_meas Loss_physics];
Loss = Loss_meas + lambda.*Loss_physics;

gradients = dlgradient(Loss,parameters);

end