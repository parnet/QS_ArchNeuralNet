function [Input] = convertInput(Input_flat)
InputT = Input_flat';
InputM = reshape(InputT,20,20,20,28);
Input = zeros(28,20,20,20);

for p = 1:28
    for m = 1:20
        for pw = 1:20
            for aw = 1:20
            Input(p,m,pw,aw) = InputM(aw,pw,m,p);
            end
        end
    end
end
end