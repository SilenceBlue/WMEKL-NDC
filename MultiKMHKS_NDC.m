function w1 = MultiKMHKS_NDC( classOne , classTwo ,b_pos_index, b_neg_index,inputInf, u)
    % input ----- classOne: matrix of numClassOne*dim
    %       ----- classTwo: matrix of numClassTwo*dim
    %       ----- b_pos_index: the index of boundary positive samples
    %       ----- b_neg_index: the index of boundary negative samples
    %       ----- inputInf: the informatiom of input parameter
    %       ----- u: the weight of kernel 
    % output ---- w1: the weight vector
    C1 = inputInf.C1;
    C2 = inputInf.C2;
    p = inputInf.p;
    len_one = size(classOne{1}, 1);
    len_two = size(classTwo{1}, 1);
    IR = len_two/len_one;
    D = diag([IR*ones(1, len_one), ones(1, len_two)]);
    len_all = len_one + len_two;
    one_all = ones(len_all, 1);
    m = inputInf.M;
    %% Parameters for Finding Boundaries and Neighbors
    k = 3; % Number of Neighbors in Finding Boundaries (0 is not select samples)
    options = [];
    options.Metric = 'Euclidean';
    options.NeighborMode = 'KNN';
    options.nk = 5; % Number of Neighbors
    options.WeightMode = 'sign';% 0-1 weighting
    for i = 1: inputInf.M
        positive = [classOne{i}, ones(size(classOne{i}, 1), 1)];
        negative = [classTwo{i}, ones(size(classTwo{i}, 1), 1)];
        if k~= 0 % Select the boundary samples 
            N = constructN(positive(b_pos_index,:), negative(b_neg_index,:), options); % Nearest Neighbor Adjacency Matrix
            Y1{i} = [positive(b_pos_index,:); negative(b_neg_index,:)];     
        else % not select the boundary samples 
            N = constructN(positive, negative, options); 
            Y1{i} = [positive; negative];
        end
        LN{i} = diag(sum(N)) - N;  % laplacian matrix
       %%  (Yw-b-I)'(Yw-b-I)      
        Y{i} = [[classOne{i}, ones(size(classOne{i}, 1), 1)]; -1*[classTwo{i}, ones(size(classTwo{i}, 1), 1)]];  
        dim = size(Y{i}, 2);
        w0{i} = 0.5*ones(dim, 1);
        b0{i} = ones(len_all, 1)*inputInf.B(i);
        I = eye(dim);
        I(end, end) = 0;
        P{i} = pinv((u(i) + C2*(1+m*(u(i)^2)-2*u(i)))*Y{i}'*D*Y{i} + u(i)*C1*I + u(i)*p*Y1{i}'*LN{i}*Y1{i}); 
    end
    %% Iterative solution of s
    [L0, weight_mean_out, b1] = getL(w0, b0, Y, Y1, one_all, D, LN, inputInf, C1, C2, p, u);
    b0 = b1;
    iter = 1;
    while iter <= inputInf.sizeIter 
        iter = iter + 1;
        for i = 1:inputInf.M
           w1{i} = P{i}*Y{i}'*D*(b0{i} + one_all + C2*(weight_mean_out - Y{i}*w0{i})/inputInf.M);
        end
        w0 = w1;
        [L1, weight_mean_out, b1] = getL(w0, b0, Y, Y1, one_all, D, LN, inputInf, C1, C2, p, u);
        if (L1 - L0)'*(L1 - L0) <= inputInf.termination 
            break;
        end
        L0 = L1;
        b0 = b1;
    end
end

function [L, weight_mean_out, b1] = getL(w, b, Y, Y1, one_all, D, LN, inputInf, C1, C2, p, u)
    left = 0;
    weight_mean_out = 0;
    for i = 1:inputInf.M
        temp = u(i)*((Y{i}*w{i} - one_all - b{i})'*D*(Y{i}*w{i} - one_all - b{i}) + C1*w{i}'*w{i} + p*(Y1{i}*w{i})'*LN{i}*(Y1{i}*w{i}));
        left = left + temp;
        weight_mean_out = weight_mean_out + u(i)*Y{i}*w{i};
        e{i} = Y{i}*w{i} - b{i} - one_all;
        b1{i} = b{i} + 0.99*(e{i} + abs(e{i}));
    end
    right = 0;
    for i = 1:inputInf.M
        temp = C2*(Y{i}*w{i} - weight_mean_out)'*(Y{i}*w{i} - weight_mean_out);
        left = left + temp;
    end
    L = left + right;
end
