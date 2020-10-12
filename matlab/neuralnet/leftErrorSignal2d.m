function RES = leftErrorSignal2d(ErrorSignal, Conv, Trace)
slp = 0.01;
[m,n] = size(ErrorSignal);
RES = zeros(2*m,2*n);
for i = 1:m
    for j = 1:n
        scale = 1;
        if(Trace(i,j) == 1)
            if(Conv(2*i-1,2*j-1) < 0)
                scale = 0.01;
            end
            RES(2*i-1,2*j-1) = scale*ErrorSignal(i,j);
        elseif(Trace(i,j) == 2)
            if(Conv(2*i-1,2*j) < 0)
                scale = 0.01;
            end
                RES(2*i-1,2*j) = scale*ErrorSignal(i,j);
        elseif(Trace(i,j) == 3)
            if(Conv(2*i,2*j-1) < 0)
                scale = 0.01;
            end
                RES(2*i,2*j-1) = scale*ErrorSignal(i,j);
        else
            if(Conv(2*i,2*j) < 0)
                scale = 0.01;
            end
                RES(2*i,2*j) = scale*ErrorSignal(i,j);
        end
    end
    
end

end