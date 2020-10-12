function RES = rightErrorSignal2d(ErrorSignal, Conv, Trace)
[m,n] = size(ErrorSignal);
RES = zeros(2*m,2*n);
for i = 1:m
    for j = 1:n

        if(Trace(i,j) == 1)
            RES(2*i-1,2*j-1) = scale*ErrorSignal(i,j);
        elseif(Trace(i,j) == 2)
                RES(2*i-1,2*j) = scale*ErrorSignal(i,j);
        elseif(Trace(i,j) == 3)
                RES(2*i,2*j-1) = scale*ErrorSignal(i,j);
        else
                RES(2*i,2*j) = scale*ErrorSignal(i,j);
        end
    end
    
end

end