function v = restrictdropout(input,dropoutindex)
[m,n]= size(dropoutindex');
v = zeros(m,1);
for i = 1:m
    v(i) = input(dropoutindex(i)+1);
end
end