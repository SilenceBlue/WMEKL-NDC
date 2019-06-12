function [ vec_res, t_train ] = MultiKMHKS_Fuc_NDC( trainSet , testSet, inputInf)
% input ----- trainSet: cell of 1*numClass
%       ----- testSet: matrix of numTest*dim
%       ----- C1: parameter
%       ----- C2: parameter
%       ----- inputInf - the informatiom of input parameter:
%                   C1: parameter
%                   C2: parameter
%                   p: parameter
%                   M: the number of kernel
%                   kType: the type of kernel
%                   sizeIter: Number of iterations (default 100)
%                   termination: Iterative termination conditions (default 1e-3)
%                   B: Initialization margin B (default 1e-6)
%                   R: learning rate (default 0.99)
% output ---- vec_res: vector of result
%        ---- t_train: traing time
C1 = inputInf.C1;
C2 = inputInf.C2;
p = inputInf.p;
totalClass = size(trainSet , 2) ;
    [lenTest , dim] = size(testSet) ;
    testLabel = testSet(:,dim) ;
    resultMat = zeros(lenTest , totalClass) ; 
    t_train = 0 ;
    % One-to-one voting (Binary and multiple classification problems are available)
    for i = 1 : totalClass
        classOne = trainSet{i} ;
        for j = i +1 : totalClass 
            classTwo = trainSet{j} ;
            tic;
            k = 3;% Selecting k Neighbors of Boundary Samples
            [b_pos_index, b_neg_index] = get_neg_boundary(classOne, classTwo, k);% get negative boundary samples
            [class_one , class_two , testData, A] = GenerateEmpiricalData(classOne , classTwo , testSet, inputInf);% get empirical mapped samples
            u = A/sum(A);
            
            w = MultiKMHKS_NCA(class_one , class_two, b_pos_index, b_neg_index, C1, C2, inputInf, p, u) ;
            t = toc;
            t_train = t_train + t;
            [temp, tPre] = class4test(w, testData, inputInf, lenTest, u);

            indexClassOne = find(temp == 1) ;% Find the location determined to be the first category
            resultMat(indexClassOne , i) = resultMat(indexClassOne , i) + 1 ;
            indexClassTwo = find(temp == -1) ;
            resultMat(indexClassTwo , j) = resultMat(indexClassTwo , j) + 1 ;
        end
    end    
    [C finalClass] = max((resultMat'));
    % Two kinds of imbalance problems (positive class label: 1, negative class label: 2)
    len_classOne = length(find(testLabel == 1));
    len_classTwo = length(find(testLabel ~= 1));
    TP = length(find(finalClass(1:len_classOne)==1));
    TN = length(find(finalClass(len_classOne+1:end)==2));
    FN = len_classOne - TP;
    FP = len_classTwo - TN;
    Acc1 = (TP+TN)/(TP+TN+FP+FN)*100;
    
    TP_rate = TP/(TP+FN);
    FP_rate = FP/(FP+TN);
    TN_rate = TN/(FP+TN);
    FN_rate = FN/(TP+FN);
    AA = (TP_rate+TN_rate)*50;% Average Accuracy
    GM = sqrt(TP_rate*TN_rate)*100;% Geometric Mean
    F1 = (2*TP)/(2*TP+FP+FN);
    tPre = 1./(1+exp(-tPre));
    [~,~,~,AUC] = perfcurve(testLabel,tPre,'1');
    AUC = AUC*100; % AUC
    vec_res = [TP_rate,TN_rate,Acc1,AA,GM,F1,AUC];
end
function [class_one , class_two , testData, A] = GenerateEmpiricalData(org_one , org_two , testSet , inputInf)
    M = inputInf.M ;  
    class_one=cell(M , 1) ;
    class_two=cell(M , 1) ;
    testData = cell(M , 1) ;
    tempKPar = aveRBFPar([org_one ; org_two] , size([org_one ; org_two] , 1)) ;
    inputInf.kPar = inputInf.kdelta .* tempKPar ;
    
    trainData.classOne = org_one ;
    trainData.classTwo = org_two ;
    t_train = 0 ;
    A = zeros(M,1);
    for i = 1 : M ;
        kernelType = char(inputInf.kType);        
        [emp_train , emp_Test , t, align] = kernel_mapping(trainData , testSet(:, 1:end - 1), kernelType , inputInf.kPar(i)) ;
        A(i) = align;
        t_train = t_train + t ;
        
        class_one(i) = {emp_train.emp_classOne} ;
        class_two(i)= {emp_train.emp_classTwo} ;
        testData(i) = {emp_Test} ;
    end
end

function [temp, pre] = class4test(w, testData, inputInf, lenTest, u)
    test_set = zeros(lenTest, 1);
    for i = 1: inputInf.M
        data = [testData{i}, ones(lenTest, 1)];
        test_set = test_set + data*w{i}*u(i);
    end
    pre = test_set;
    temp = sign(test_set); 
    temp(find(temp == 0)) = 1;
end

% Get the parameter delta of RBF kernel
function par=aveRBFPar(data , size)
    mat_temp = sum(data.^2,2) * ones(1,size) + ones(size,1)*sum(data.^2,2)' - 2* data*data';
    tempMean = (1/size^2) * sum(sum(mat_temp,1),2) ;
    par = sqrt(tempMean) ;
end
