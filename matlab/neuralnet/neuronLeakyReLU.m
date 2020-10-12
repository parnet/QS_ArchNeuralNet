function v = neuronLeakyReLU(w)
[m,n] = size(w);
v = zeros(m,n);
for i = 1:m
    if (w(i) < 0)
        v(i) = 0.01* w(i);
    else
        v(i) = w(i);
    end 
end
end