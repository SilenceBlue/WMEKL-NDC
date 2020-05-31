# WMEKL-NDC
Weight-Based Multiple Empirical Kernel Learning with Neighbor Discriminant Constraint for Heart Failure Mortality Predicrion

The code is developed under Matlab 2015b

For the specific use of each function, please see the comments for the function.
## Demo Code
inputInf.C1 = 0.1;

inputInf.C2 = 0.1;

inputInf.p = 0.1;

inputInf.M = 3;

inputInf.kType = 'rbf';

inputInf.sizeTter = 100; 

inputInf.termination = 1e-3;

inputInf.R = 0.99/*ones(inputInf.M, 1);

inputInf.B = 0.1/*ones(inputInf.M, 1);

[vec_res, t_train] = MultiKMHKS_Fuc_NDC(train, test, inputInf);

TPR = vec_res(1);

TNR = vec_res(2);

Acc = vec_res(3);

AA = vec_res(4);

GM = vec_res(5);

F1 = vec_res(6);

AUC = vec_res(7);
