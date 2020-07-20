
function num = find_num(filenames)
% Given a list of files, find the number of tr**.mat files.
num = -1;
for n = 1 : size(filenames,1)
    name = filenames(n).name;
    a = strsplit(name,'.');
    if strcmp(a{end},['hdr'])
        b = strsplit(a{1},'tr');
        if isempty(b{1})
            num = max(num,str2num(b{2}));
        end
    end
end
end
