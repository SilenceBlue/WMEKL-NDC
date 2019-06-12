function [emp_train , emp_Test , t_train, align] = kernel_mapping(train_data , test_data , kernelPerType , kernelPar)
    % Get the mapped train_data and test_data
    %
    % input ----- train_data = {train_class_one , train_class_two } 
    %       ----- test_data = [X_in_input_space ] 
    %       ----- kernelPerType: the type of kernel
    %       ----- kernelPar: the parameter of kernel
    % output ---- emp_train: the empirical mapped train_data
    %        ---- emp_Test: the empirical mapped test_data
    %        ---- t_train: training time
    %        ---- align: the value of KTA
    % 
    %
    
    train_class_one = train_data.classOne ;
    train_class_two = train_data.classTwo ;
    len_ClassOne = size(train_class_one , 1) ;
    len_ClassTwo = size(train_class_two , 1) ;
    
    target = [ones(len_ClassOne,1);-1*ones(len_ClassTwo,1)];
    targetKernel = target*target';
    [emp_trn_all , emp_Test , t_train, align] = emp_Generator([train_class_one ; train_class_two] , test_data , kernelPerType , kernelPar, targetKernel) ;%产生映射后的训练，测试集

    emp_classOne = emp_trn_all(1:len_ClassOne , :) ;
    emp_classTwo = emp_trn_all(len_ClassOne+1 : len_ClassOne+len_ClassTwo , :);
    emp_train.emp_classOne = emp_classOne ;
    emp_train.emp_classTwo = emp_classTwo ;
    
    clear temp_emp emp_classOne emp_classTwo ;
end

function [emp_train , emp_Test , t_train, align] = emp_Generator(trainData , testData , kType , kPar, targetKernel)
    % Training set and test set mapping together
    % start clock for trainData
    tic  
    implicitKernel = Kernel(trainData , trainData , kType , kPar) ;
    align = CKA(implicitKernel, targetKernel);
    [pc , variances , explained] = pcacov(implicitKernel);%[PC, LATENT, EXPLAINED] = pcacov(X)

    i = 1 ;
    label = 0 ;
    while variances(i) >= 1e-3 ;
        if i+1 > size(variances,1) ;
            label = 1 ;
            break ;
        end;
        i = i + 1 ;    
    end;

    if label == 0 ;
        i = i - 1 ;
    end;

    index = 1 : i ;
    P = pc(: , index) ;
    R = diag(variances(index)) ;
    emp_train = implicitKernel * P * R^(-1/2) ;% ekm=KAQ.^(-1/2)      
    t_train = toc ;
    
    kerTestMat = Kernel(testData ,trainData , kType , kPar) ;
    emp_Test = kerTestMat * P * R^(-1/2) ;  
end

