function dB = biasChange3d(leftErrorSignal)
    dB = sum(sum(sum(leftErrorSignal)));
end