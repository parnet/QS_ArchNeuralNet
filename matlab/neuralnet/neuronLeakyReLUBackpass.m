function v = neuronLeakyReLUBackpass(inp,err)
[m,n] = size(inp);
v = zeros(m,n);
for i = 1:m
    if(inp(i) < 0)
        v(i) = err(i) * 0.01;
    else
        v(i) = err(i);
    end
end
end