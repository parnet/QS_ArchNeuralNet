function v = neuronSoftmax(w)
[m,n] = size(w);
v = zeros(m,n);

% w = neuronSoftmaxMinTranslation(w);

sum = 0;
for i = 1:m
    sum = sum + exp(w(i));
end 

% disp(sum)

for i = 1:m
    v(i) = exp(w(i))/sum;
end 
end