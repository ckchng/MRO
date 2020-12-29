function [RotsAvg, iter, objVals] = rotavg(Rots, R)
% rotation averaging

objVals = [];

R0 = [Rots{:}];
Y0 = R0'*R0;
Y = Y0;

m = length(Rots);

% Solve rot avg from R and Y0

oldObj = -trace(R*Y);
numOfIters = 10000;

for iter = 1:numOfIters % add convergence criteria
    % select k
    %k = randsample(1:m, 1);
    for k = 1:m
        % B: eliminating the kth row and column from Y
        B = Y;
        B(3*k-2:3*k,:) = []; % eliminating kth row
        B(:,3*k-2:3*k) = []; % eliminating kth col
        
        % W: eliminating the kth col and all but the kth row from R
        W = R(3*k-2:3*k, :); % the kth row from R
        W(:,3*k-2:3*k) = []; % eliminating kth col
        W = W'; % make W a column vector
        
        % Obtain S
        S = B*W*pinv( sqrtm(W'*B*W) ); 
        
        Y = [eye(3) S'; S B];
        
        % reorder
        order = [4:3*k 1:3 (3*k+1):3*m];
        Y = Y(order,order);
    end
    
    % eval soluiton
    obj = -trace(R*Y);
    
    objVals(end+1)=obj;
    
    fprintf('obj    %f    oldObj  %f\n', obj, oldObj);
    
     if (oldObj-obj)/max(abs(oldObj),1) <= 1e-8
         break
     end
    oldObj = obj;
end
%
% Obtain Ropt
RotsAvg = cell(1,m);
for i = 1:m
    RotsAvg{i} = Y(1:3, 3*i-2:3*i);
    
    if det(RotsAvg{i})<0
        RotsAvg{i} = -RotsAvg{i};
    end
end
end