function Pool = showPooling(Pooling,i,j,k)

Pool = zeros(2,2,2);
Pool(1,1,1) = Pooling(i,j,k);
Pool(1,1,2) = Pooling(i,j,k+1);
Pool(1,2,1) = Pooling(i,j+1,k);
Pool(1,2,2) = Pooling(i,j+1,k+1);

Pool(2,1,1) = Pooling(i+1,j,k);
Pool(2,1,2) = Pooling(i+1,j,k+1);
Pool(2,2,1) = Pooling(i+1,j+1,k);
Pool(2,2,2) = Pooling(i+1,j+1,k+1);

end