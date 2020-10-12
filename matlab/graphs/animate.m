function [] = animate(M)
h = figure;
axis tight manual % this ensures that getframe() returns a consistent size
filename = 'testAnimated_x.gif';
for n = 1:1:20
    % Draw plot for y = x.^n
    imagesc(reshape(M(:,:,n,1),20,20));
    drawnow 
      % Capture the plot as an image 
      frame = getframe(h); 
      im = frame2im(frame); 
      [imind,cm] = rgb2ind(im,256); 
      % Write to the GIF File 
      if n == 1 
          imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
      else 
          imwrite(imind,cm,filename,'gif','WriteMode','append'); 
      end 
end
end