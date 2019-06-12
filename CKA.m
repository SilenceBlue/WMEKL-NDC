function align = CKA( implicitKernel, targetKernel )
    % Centered Kernel alignment
    dim = size(implicitKernel,2);
    one = ones(dim,1);
    I = eye(dim);
    c_implicitKernel = (I - (one*one')/dim)*implicitKernel*(I - (one*one')/dim);
    c_targetKernel = (I - (one*one')/dim)*targetKernel*(I - (one*one')/dim);
    align = trace(c_implicitKernel'*c_targetKernel)/...
            sqrt(trace(c_implicitKernel'*c_implicitKernel)*trace(c_targetKernel'*c_targetKernel));
end

