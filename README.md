# WMEKL-NDC
Weight-Based Multiple Empirical Kernel Learning with Neighbor Discriminant Constraint for Heart Failure Mortality Predicrion
The code is developed under Matlab 2015b
# Demo Code
inputInf.C1 = 0.1;
inputInf.C2 = 0.1;
inputInf.p = 0.1;
inputInf.M = 3;
inputInf.kType = 'rbf';
inputInf.sizeTter = 100; 
inputInf.termination = 1e-3;
inputInf.R = 0.99/*ones(inputInf.M, 1);
inputInf.B = 0.1/*ones(inputInf.M, 1);
[vec_res, t_train] = MultiKMHKS_Fuc_NCA(train, test, inputInf);
