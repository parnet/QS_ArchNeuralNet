function [] = dispim(K)
imagesc(K);
hold on;
line([1,1], [100,100], 'Color', 'r');
end