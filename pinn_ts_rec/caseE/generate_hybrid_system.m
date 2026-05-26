function [Lorenz,LV] = generate_hybrid_system()

t_start = 0;
t_end = 30;
dt = 1e-5;

t = t_start:dt:t_end;

%% Lorenz

sigma = 10;
rho = 28;
beta = 8/3;

x = zeros(size(t));
y = zeros(size(t));
z = zeros(size(t));

x(1) = 5;
y(1) = 2;
z(1) = 30;

%% Simulation

for i = 2 : length(t)
    
    x(i) = x(i-1) + sigma*(y(i-1)-x(i-1))*dt;
    y(i) = y(i-1) + x(i-1)*(rho-z(i-1))*dt - y(i-1)*dt;
    z(i) = z(i-1) + (x(i-1)*y(i-1)-beta*z(i-1))*dt;

end


%%

Lorenz.t = t;
Lorenz.x = x;
Lorenz.y = y;
Lorenz.z = z;

Lorenz.beta = beta;
Lorenz.sigma = sigma;
Lorenz.rho = rho;

Lorenz.dt = dt;

%% Lotka-Volterra Parameters and Initial Conditions

a = 10;
b = 2;
c = 3;
d = 4;

x = zeros(size(t));
y = zeros(size(t));

x(1) = 0.5;
y(1) = 0.5;

%%

K = 100;

t_num = 0;

for i = 2 : length(t)

    x0 = x(i-1);
    y0 = y(i-1); 

    dt_num = 0;

    for k = 1 : K

        dt_sim = dt/K;

        x1 = x0 + (a*x0 - b*x0*y0)*dt_sim;
        y1 = y0 + (c*x0*y0 - d*y0)*dt_sim;

        x0 = x1;
        y0 = y1;

        dt_num = dt_num + dt_sim;

    end

    t_num(i) = t_num(i-1) + dt_num;
    x(i) = x1;
    y(i) = y1;

end


LV.t = t;
LV.x = x;
LV.y = y;
LV.a = a;
LV.b = b;
LV.c = c;
LV.d = d;
LV.dt = dt;

end