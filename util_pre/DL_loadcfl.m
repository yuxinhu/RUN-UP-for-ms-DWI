function [k, sens,im, b0] = DL_loadcfl(filepath, nx, ny, nc, ns)
if nargin < 5
    ns = 4;
end
if nargin < 4
    nc = 8;
end
if nargin < 3
    ny = 256;
end
if nargin < 2
    nx = 256;
end

a1 = nx * ny * nc * ns;
a2 = nx * ny * nc;
a3 = nx * ny * ns;

data = readcfl([filepath]);
k = permute(reshape(data(1:a1),[nx ny ns nc]),[1 2 4 3]); % nx - ny - nc - ns
sens = reshape(data(a1+1:a1+a2),[nx ny nc]);
im = reshape(data(a1+a2+1:a1+a2+a3),[nx ny ns]);
b0 = reshape(data(a1+a2+a3*2+1 :a1+a2+a3*2+nx*ny), [nx ny]);
end


