function RES = leftErrorSignal3d(ErrorSignal, Conv, Trace)
slp = 0.01;
[m,n,p] = size(ErrorSignal);
RES = zeros(2*m,2*n,2*p);
for i = 1:m
    for j = 1:n
        for k = 1:p
        scale = 1;
        if(Trace(i,j,k) == 1)
            if(Conv(2*i-1,2*j-1,2*k-1) < 0)
                scale = 0.01;
            end
            RES(2*i-1,2*j-1,2*k-1) = scale*ErrorSignal(i,j,k);
        elseif(Trace(i,j,k) == 2)
            if(Conv(2*i-1,2*j,2*k-1) < 0)
                scale = 0.01;
            end
                RES(2*i-1,2*j,2*k-1) = scale*ErrorSignal(i,j,k);
        elseif(Trace(i,j,k) == 3)
            if(Conv(2*i,2*j-1,2*k-1) < 0)
                scale = 0.01;
            end
                RES(2*i,2*j-1,2*k-1) = scale*ErrorSignal(i,j,k);
        elseif(Trace(i,j,k) == 4)
            if(Conv(2*i,2*j,2*k-1) < 0)
                scale = 0.01;
            end
                RES(2*i,2*j,2*k-1) = scale*ErrorSignal(i,j,k);
        elseif(Trace(i,j,k) == 5)
            if(Conv(2*i-1,2*j-1,2*k) < 0)
                scale = 0.01;
            end
            RES(2*i-1,2*j-1,2*k) = scale*ErrorSignal(i,j,k);
        elseif(Trace(i,j,k) == 6)
            if(Conv(2*i-1,2*j,2*k) < 0)
                scale = 0.01;
            end
                RES(2*i-1,2*j,2*k) = scale*ErrorSignal(i,j,k);
        elseif(Trace(i,j,k) == 7)
            if(Conv(2*i,2*j-1,2*k) < 0)
                scale = 0.01;
            end
                RES(2*i,2*j-1,2*k) = scale*ErrorSignal(i,j,k);
        elseif(Trace(i,j,k) == 8)
            if(Conv(2*i,2*j,2*k) < 0)
                scale = 0.01;
            end
                RES(2*i,2*j,2*k) = scale*ErrorSignal(i,j,k);
        else
        end
        
        end
    
end

end