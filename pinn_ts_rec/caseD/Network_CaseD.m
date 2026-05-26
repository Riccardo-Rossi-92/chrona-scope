%% Net 01

function [X,parameters] = Network_CaseD(X,Predict,parameters,Network)

Layer = Network.Layer;

if Predict == 0
    
    %% Initialise net

    parameters = [];

    parameters.scale.t0 = dlarray(mean(Network.time_window));
    parameters.scale.dt = dlarray(Network.time_window(2)-Network.time_window(1));

    X = (X - parameters.scale.t0)./parameters.scale.dt;

    
    for i = 1 : length(Layer)

        parameters.("p"+i).weights = dlarray(randn([Layer(i) size(X,1)])*sqrt(2/Layer(i)));
        parameters.("p"+i).bias = dlarray(zeros([Layer(i) 1]));

        X = fullyconnect(X,parameters.("p"+i).weights,parameters.("p"+i).bias);
        X = tanh(X);

    end

    parameters.output.weights = dlarray(randn([3 size(X,1)]))/3;
    parameters.output.bias = dlarray(zeros([3 1]));

    X = fullyconnect(X,parameters.output.weights,parameters.output.bias);

    parameters.scale.X = dlarray([Network.ScaleX; Network.ScaleY; Network.ScaleZ]); 

    X = X.*parameters.scale.X;

    %% 

    parameters.param.sigma = dlarray(10);
    parameters.param.rho = dlarray(10);
    parameters.param.beta = dlarray(10);
    parameters.param.alpha1 = dlarray(10);
    parameters.param.alpha2 = dlarray(10);
    parameters.param.alpha3 = dlarray(10);

else

    X = (X - parameters.scale.t0)./parameters.scale.dt;

    for i = 1 : length(Layer)
        X = fullyconnect(X,parameters.("p"+i).weights,parameters.("p"+i).bias);
        X = tanh(X);
    end

    X = fullyconnect(X,parameters.output.weights,parameters.output.bias);

    X = X.*parameters.scale.X;

end

end


