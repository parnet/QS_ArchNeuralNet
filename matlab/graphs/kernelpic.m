function M = kernelpic(A)
    M = zeros(3*3*10,4*3*2*7);
    
    for pol = 1:10
        for part = 1:14
            for f = 1:4
                for i = 1:3
                    for j = 1:3
                        for k = 1:3
                            u = k;
                            v = j;
                            w = i;
                            x = v + (u-1)*3+9*(pol-1);
                            y = w+3*(part-1)+(f-1)*3*14;
                            M(x,y) = A((pol-1)*4+f,(part-1)*27+(i-1)*9+(j-1)*3+k);
                        end
                    end
                end
            end
        end
    end
        
end
    
