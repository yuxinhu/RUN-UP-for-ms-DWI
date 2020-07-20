function [p,p2,p3] = cal_value(im1,im2)
im1 = im1(:,:,:);
im2 = im2(:,:,:);
for i =1 : size(im1,3)
    p(i) = psnr(im1(:,:,i),im2(:,:,i),max(max(im2(:,:,i))));
    p2(i) = NRMSE(im1(:,:,i),im2(:,:,i));
    p3(i) = ssim(im1(:,:,i),im2(:,:,i));
end

end

