function Wx = nonHiddenBoth(W,nqgp,qgp)
dN = noneHidden(W,nqgp,"nqgp");
dQ = noneHidden(W,qgp,"qgp");
Wx = W - 0.001*(dN+dQ);
end