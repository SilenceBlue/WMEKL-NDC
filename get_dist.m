function [D] = get_dist(X,Y,n_X,n_Y)   

% ��X��Y������������������ĺ�����Ҫ�������ڲ���������������X=Y
% X,Y��һ��һ���������ޱ��
% n_X��n_Y��X��Y����
% D��n_X����n_Y�еľ������

    % ��ŷ�Ͼ���
        D_temp = sum(X.^2,2)*ones(1,n_Y) + ones(n_X,1)*sum(Y.^2,2)' - 2*X*Y'; %||xi-xj||2
        D = sqrt(D_temp);
            
end%functionb
