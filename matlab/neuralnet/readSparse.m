function A = readSparse(path)
raw = importdata(path);
A = zeros(224000,1);
[~,n] = size(raw);

% skip num of entries
% skip num of particles
for i = 3:2:n
    A(raw(i)+1) = raw(i+1);
end
end