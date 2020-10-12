function lerr = neuronSoftmaxBackpass(in, err)
[m, n] = size(in);
lerr = zeros(m,n);
sum = 0;
for i = 1:m
    sum = sum + exp(in(i));
end

for i = 1:m
    act_i = exp(in(i))/sum;
    for j = 1 :m
        act_j = exp(in(j))/sum;
        delta_ij = 0;
        if(i == j)
            delta_ij = 1;
        end
        lerr(i) = lerr(i) + err(j)*(act_i*(delta_ij - act_j));
    end
end
end