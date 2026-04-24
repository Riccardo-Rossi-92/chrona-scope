function [X,t,info] = LorenzAttractor(t_end,N,X0,method,param)
% LorenzAttractor simulates the Lorenz system
% [X,t,info] = LorenzAttractor(t_end,N,X0,method,param)
% t_end: end time
% N: number of time steps
% X0: initial condition [x0,y0,z0]
% method: "Euler" or "ODE45"
% param: [sigma; rho; beta]

%% Set defaults
if nargin < 5 || isempty(param)
    param = [10; 28; 8/3];
end
if nargin < 4 || isempty(method)
    method = "Euler"; 
end
if nargin < 3 || isempty(X0)
    X0 = [1,1,1];
end
if nargin < 2 || isempty(N)
    N = 1000;
end
if nargin < 1 || isempty(t_end)
    t_end = 10;
end

%% Time vector
t = linspace(0, t_end, N);
dt = mean(diff(t));

%% Preallocate
X = zeros(N,3);
X(1,:) = X0;

%% Parameters
sigma = param(1);
rho   = param(2);
beta  = param(3);

%% Simulation
switch method
    case "Euler"
        for i = 2:N
            X(i,1) = X(i-1,1) + sigma*(X(i-1,2)-X(i-1,1))*dt;
            X(i,2) = X(i-1,2) + (X(i-1,1)*(rho - X(i-1,3)) - X(i-1,2))*dt;
            X(i,3) = X(i-1,3) + (X(i-1,1)*X(i-1,2) - beta*X(i-1,3))*dt;
        end
    case "ODE45"
        f = @(~,a) [-sigma*a(1) + sigma*a(2);
                     rho*a(1) - a(2) - a(1)*a(3);
                     -beta*a(3) + a(1)*a(2)];
        [t_ode,X_ode] = ode45(f,[0 t_end],X0);
        % Interpolate to uniform time vector
        X = interp1(t_ode,X_ode,t,'linear');
    otherwise
        error('Unknown method. Use "Euler" or "ODE45".');
end

%% Info
info.method = method;
info.param = param;
info.dt = dt;

end