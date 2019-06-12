function [ pos_index, neg_index ] = get_boundary( positive, negative, k )
% get boundary samples
    len_one = size(positive, 1);
    len_two = size(negative, 1);
    % 到正类最近的负类样本index
    negative_index= [];
    for i_pos = 1:len_one
        vec_dist = get_dist(positive(i_pos,:), negative, 1, len_two);
        [~, pos_near_index] = sort(vec_dist);
        negative_index = [negative_index;pos_near_index(1:k)];
    end         
    neg_index = unique(negative_index);
    % 到被选出的负类样本的正类样本index
    positive_index = [];
    for i_neg = 1:len_two
        vec_dist = get_dist(negative(i_neg,:), positive, 1, len_one);
        [~, neg_near_index] = sort(vec_dist);
        positive_index = [positive_index;neg_near_index(1:k)];
    end   
    pos_index = unique(positive_index);    
end

