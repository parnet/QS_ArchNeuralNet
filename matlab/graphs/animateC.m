function [] = animateC(M)
h = figure;
axis tight manual % this ensures that getframe() returns a consistent size
filename = 'animate_inclination.gif';
for n = 1:1:24
    % Draw plot for y = x.^n
    imagesc(reshape(M(n,:,:),24,24));
    colorbar();
    drawnow 
      % Capture the plot as an ima
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