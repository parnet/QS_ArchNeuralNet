function v = dropout(input, index, dropout)
[n,m] = size(index);
v = zeros(m,n);
scaling = 1.0/(1-dropout);

for i = 1:m
    v(i) = input(index(i)+1) * scaling;
end
end