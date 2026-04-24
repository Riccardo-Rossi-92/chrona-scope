function D2 = CorrelationDimension_Krakosvka(X)
%
% Correlation dimension estimator of a set X of points that is extremely simple because
% it evaluates only the two nearest neighbors of the points within the set being examined.
%
% The method has been introduced in:
%       Krakovská, Anna and Chvosteková, Martina.
%       "Simple correlation dimension estimator and its use to detect causality."
%       Chaos, Solitons & Fractals 175 (2023): 113975
%
%  X:    The input data matrix (T x D), where T represents length of data and D is the dimension (number of vector coordinates)  
%  D2:   The output - estimation of the correlation dimension of X
%

    T = size(X,1); % Number of data points
    % idx, D
    [idx,D] = knnsearch(X,X,'K',floor(0.1*T)+10,'NSMethod','kdtree','Distance','euclidean'); %'chebychev');
    % Initialize an array to store the local estimations
    D_local = nan(1,T);    
    for i = 1:T   
        pom1 = 2;
        i1 = idx(i,pom1);
        D1 = D(i,pom1);
        while D1 == 0
            pom1 = pom1 + 1;
            i1 = idx(i,pom1);
            D1 = D(i,pom1);
        end
        data_idx1 = X(i1,:); 
        pom2 = pom1 + 1;
        i2 = idx(i,pom2);
        while D(i,pom1) == D(i,pom2)
            pom2 = pom2 + 1;
            i2 = idx(i,pom2);
        end            
        data_idx2 = X(i2,:); 
        % Calculate distance between the 2 neighbors                            
        D12 = pdist([data_idx1; data_idx2]);
        if D12 <= D(i,pom1) 
            D_local(i) = log(3/2)/log(D(i,pom2)/D(i,pom1));  
        elseif D(i,pom1) < D12 && D12 <= D(i,pom2)
            D_local(i) = log(3)/log(D(i,pom2)/D(i,pom1));
        else
            D_local(i) = log(2)/log(D(i,pom2)/D(i,pom1));
        end
    end         
    
    D2 = median(D_local,'omitnan');
end
