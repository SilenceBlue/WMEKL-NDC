function [ pos_index, neg_index ] = get_neg_boundary( positive, negative, k )
    % get boundary samples
    len_one = size(positive, 1);
    len_two = size(negative, 1);
    % The index of nearest negative sample to positive class
    negative_index= [];
    for i_pos = 1:len_one
        vec_dist = get_dist(positive(i_pos,:), negative, 1, len_two);
        [~, pos_near_index] = sort(vec_dist);
        negative_index = [negative_index;pos_near_index(1:k)];
    end         
    neg_index = unique(negative_index);
    pos_index = [1:len_one]';    
end

