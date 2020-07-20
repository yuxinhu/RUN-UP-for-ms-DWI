function res = NRMSE(im1,im2)
% Given two 2D images, calculate the normalized root sum of squared error.
res = abs(im1(:)-im2(:));
res = res.^2;
res = sqrt(sum(res)/length(res))/(max(abs(im2(:))) - min(abs(im2(:))));
end