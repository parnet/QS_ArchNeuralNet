function [Output,Trace] = pooling2d(Active)
[m,n] = size(Active);
Output = zeros(m/2,n/2);
Trace = zeros(m/2,n/2);

for i = 1: m/2
    for j = 1: n/2
        [Val,Index] = max([Active(2*i-1,2*j-1) Active(2*i-1,2*j) Active(2*i,2*j-1) Active(2*i,2*j)]);
        Trace(i,j) = Index;
        Output(i,j) = Val;
    end
end