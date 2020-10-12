function v = antidropout(input, index, fullsize)
v = zeros(fullsize, 1);
[m,n] = size(index');

for i = 1:m
    v(index(i)+1) = input(i);
end

end