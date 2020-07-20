function [llr] = fast_llr(ktemp_cc,sens, lambda, iter)
% Do shot-LLR reconstruction on preprocessed data (tr***.mat).
% ktemp_cc: nx-ny-nc-nshot
% sens: nx-ny-nc

if nargin < 4
    iter = 200;
end
if nargin < 3
    lambda = 0.002;
end


ktemp = permute(ktemp_cc,[1 2 5 3 6 4]);
sens2 = permute(sens,[1 2 4 3]);

comm = sprintf(['llr = squeeze(bart(',char(39),...
                'pics -R L:7:7:%d -w 1 -i %d',char(39),', ktemp,sens2));'], lambda,iter);
eval(comm);

end

