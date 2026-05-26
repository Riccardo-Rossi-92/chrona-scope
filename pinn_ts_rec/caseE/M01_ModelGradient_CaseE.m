function [gradients,Loss,Losses] = M01_ModelGradient_CaseE(parameters,Network,...
    dtp,dtm,dxm,dym,dzm,lambda,sx,sy,sz)


%% Coefficients

% gamma = 2;
sigma = 10;
rho = 28;
beta = 8/3;
a = 10;
b = 2;
c = 3;
d = 4;

% sigma = parameters.param.sigma;
rho = parameters.param.rho;
beta = parameters.param.beta;
% a = parameters.param.a;
% b = parameters.param.b;
c = parameters.param.c;
d = parameters.param.d;

%% Measurements

X =  Network_CaseE(dtm,1,parameters,Network);

dxp = X(1,:) + X(4,:);
dyp = X(2,:) + X(5,:);
dzp = X(3,:);

Z2x = ((dxp-dxm)/sx).^2;
Z2y = ((dyp-dym)/sy).^2;
Z2z = ((dzp-dzm)/sz).^2;

% Lx = Z2x.^(gamma+1)./(1+Z2x.^gamma);
% Ly = Z2y.^(gamma+1)./(1+Z2y.^gamma);
% Lz = Z2z.^(gamma+1)./(1+Z2z.^gamma);

Loss_meas = (mean(Z2x) + mean(Z2y) + mean(Z2z))/3;

%% Physics

X =  Network_CaseE(dtp,1,parameters,Network);

L_x = X(1,:);
L_y = X(2,:);
L_z = X(3,:);
LV_x = X(4,:);
LV_y = X(5,:);

L_dxdt = dlgradient(sum(L_x),dtp);
L_dydt = dlgradient(sum(L_y),dtp);
L_dzdt = dlgradient(sum(L_z),dtp);

LV_dxdt = dlgradient(sum(LV_x),dtp);
LV_dydt = dlgradient(sum(LV_y),dtp);


fL_x = L_dxdt - sigma.*(L_y-L_x);
fL_y = L_dydt - L_x.*(rho-L_z) + L_y;
fL_z = L_dzdt - L_x.*L_y + beta*L_z;
% 
% fLV_x = LV_dxdt - (a.*LV_x - b.*LV_x.*LV_y);
% fLV_y = LV_dydt - (c.*LV_x.*LV_y - d.*LV_y);

fLV_x = LV_dxdt./LV_x - (a - b.*LV_y);
fLV_y = LV_dydt./LV_y - (c.*LV_x - d);

Loss_physics = mean(fL_x.^2)./std(L_x).^2 + mean(fL_y.^2)./std(L_y).^2 + ...
    mean(fL_z.^2)./std(L_z).^2 + mean(fLV_x.^2) + ...
    mean(fLV_y.^2);

%% Loss total

Losses = [Loss_meas Loss_physics];
Loss = (Loss_meas + lambda.*Loss_physics)./(1+lambda);

gradients = dlgradient(Loss,parameters);

end