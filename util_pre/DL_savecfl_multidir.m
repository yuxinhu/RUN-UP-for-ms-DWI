function DL_savecfl_multidir(dirnames, Outpath, image_index)
% Given a list of file paths (reconstructed by joint reconstruction), this function
% output a list a .cfl files (one for each slice and direction) as training data.

% Input:
%   dirnames: a list of directory names. Each direction should have files tr***.mat and gt***.mat.
%             tr***.mat contains the k-space data, sensitivity map. gt***.mat contains corresponding
%             ground truth images (reconstructed by joint reconstruction).
%   Outpath: a path to save the cfl file.
%   image_index: the files will be indexed from this number. Default is 0.


if nargin < 3
    image_index = 0;
end
if (exist(Outpath,'dir') ~= 7)
    mkdir(Outpath);
end
for dir_index = 1 : length(dirnames)
    filenames = dir(dirnames{dir_index});
    for file_index = 1 : find_num(filenames)
        load([dirnames{dir_index},'/tr',num2str(file_index),'.mat'])
        load([dirnames{dir_index},'/gt',num2str(file_index),'.mat'])

        save_cfl(ktemp_cc,sens,repmat(im1,[1 1 size(phase1,3)]).*exp(1j*phase1),...
            image_index,Outpath);
        image_index = image_index + p.nex;
    end
    
	disp(['generating file from ',num2str(dir_index),':',dirnames{dir_index}])
    
end

end




%%%%%%%%%%%%%%%% save data into cfl %%%%%%%%%%%%%%%%%%%
function save_cfl(k, sens, im, image_index, Output)
% Save the data into a long vector in to the .cfl format.
% input
% k: nx-ny-nc-nshot-nex
% sens: nx-ny-nc
% im: nx-ny-nshot-nex
% saveinto:
% k'(1:nex): nx-ny-nshot-nc
% sens
% im'(1:nex): nx-ny-nshot
% mask: nx-ny-nshot
k = permute(k,[1 2 4 3 5]);
mask = abs(k(:,:,:,1,1)) ~= 0;
for n = 1 : size(k,5)
    a = k(:,:,:,:,n);
    a = a(:);
    a = [a;sens(:)];
    b = im(:,:,:,n);
    a = [a;b(:)];
    a = [a;mask(:)];
    writecfl([Output,'/tr',num2str(image_index+n)],a);
end

end


%%%%%%%%%%%%%%%% find #tr files in a folder %%%%%%%%%%%%
function num = find_num( filenames )
num = -1;
for n = 1 : size(filenames,1)
    name = filenames(n).name;
    a = strsplit(name,'.');
    if strcmp(a{end},['mat'])
        b = strsplit(a{1},'tr');
        if isempty(b{1})
            num = max(num,str2num(b{2}));
        end
    end
end
end

%%%%%%%%%%%%%%%%%%%% write cfl %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function writecfl(filenameBase,data)
% writecfl(filenameBase, data)
%    Writes recon data to filenameBase.cfl (complex float)
%    and write the dimensions to filenameBase.hdr.
%
%    Written to edit data for the Berkeley recon.
%
% Copyright 2013. Joseph Y Cheng.
% 2012 Joseph Y Cheng (jycheng@mrsrl.stanford.edu).

    dims = size(data);
    writeReconHeader(filenameBase,dims);

    filename = strcat(filenameBase,'.cfl');
    fid = fopen(filename,'w');
    
    data_o = zeros(prod(dims)*2,1);
    data_o(1:2:end) = real(data(:));
    data_o(2:2:end) = imag(data(:));
    
    fwrite(fid,data_o,'float32');
    fclose(fid);
end

function writeReconHeader(filenameBase,dims)
    filename = strcat(filenameBase,'.hdr');
    fid = fopen(filename,'w');
    fprintf(fid,'# Dimensions\n');
    for N=1:length(dims)
        fprintf(fid,'%d ',dims(N));
    end
    if length(dims) < 5
        for N=1:(5-length(dims))
            fprintf(fid,'1 ');
        end
    end
    fprintf(fid,'\n');
    
    fclose(fid);
end

