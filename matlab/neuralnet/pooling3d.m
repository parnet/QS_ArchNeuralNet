function [Output,Trace] = pooling3d(Active)
[m,n,p] = size(Active);
Output = zeros(m/2,n/2,p/2);
Trace = zeros(m/2,n/2,p/2);

for i = 1: m/2
    for j = 1: n/2
        for k = 1:p/2
        [Val,Index] =  max([Active(2*i-1,2*j-1,2*k-1) Active(2*i-1,2*j,2*k-1) Active(2*i,2*j-1,2*k-1) Active(2*i,2*j,2*k-1) Active(2*i-1,2*j-1,2*k) Active(2*i-1,2*j,2*k) Active(2*i,2*j-1,2*k) Active(2*i,2*j,2*k)]);
        Trace(i,j,k) = Index;
        Output(i,j,k) = Val;
        end
    end
end