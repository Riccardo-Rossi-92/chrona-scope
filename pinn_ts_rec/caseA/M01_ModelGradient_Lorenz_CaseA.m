function [gradients,Loss,Losses] = M01_ModelGradient_Lorenz_CaseA(parameters,Network,...
                                    dtp,dtm,dxm,dym,dzm,lambda,sx,sy,sz)


%% Coefficients

gamma = 2;

sigma = 10;
rho = 28;
beta = 8/3;

%% Measurements

X = Network_CaseA(dtm,1,parameters,Network);

dxp = X(1,:);
dyp = X(2,:);
dzp = X(3,:);

Z2x = ((dxp-dxm)/sx).^2;
Z2y = ((dyp-dym)/sy).^2;
Z2z = ((dzp-dzm)/sz).^2;

Lx = Z2x.^(gamma+1)./(1+Z2x.^gamma);
Ly = Z2y.^(gamma+1)./(1+Z2y.^gamma);
Lz = Z2z.^(gamma+1)./(1+Z2z.^gamma);

% Loss_meas = mean(Lx+Ly+Lz);
% Loss_meas = mean(Lx+Ly);
Loss_meas = mean(Lx);

%% Physics

X = Network_CaseA(dtp,1,parameters,Network);

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