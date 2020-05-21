generate_mat('Set5', 2);
generate_mat('Set5', 4);

generate_mat('Set14', 2);
generate_mat('Set14', 4);

generate_mat('BSDS100', 2);
generate_mat('BSDS100', 4);

clear;close all;

function generate_mat(dataset, scale)
%% settings
folder = strcat('dataset/test/', dataset);

%% generate data
filepaths = dir(fullfile(folder,'*.png'));

for i = 1 : length(filepaths)        
    im_gt = imread(fullfile(folder,filepaths(i).name));
    
    disp(fullfile(folder,filepaths(i).name));
    % modcrop
    if size(im_gt, 3) == 1
        sz = size(im_gt);
        sz = sz - mod(sz, scale);
        im_gt = im_gt(1:sz(1), 1:sz(2));
    else
        tmpsz = size(im_gt);
        sz = tmpsz(1:2);
        sz = sz - mod(sz, scale);
        im_gt = im_gt(1:sz(1), 1:sz(2),:);
    end
    
    if size(im_gt, 3) == 1
        im_gt = cat(3, im_gt, im_gt, im_gt);
    end
    
    im_gt = double(im_gt);
    im_gt_ycbcr = rgb2ycbcr(im_gt / 255.0);
    im_gt_y = im_gt_ycbcr(:,:,1) * 255.0;
    im_l_ycbcr = imresize(im_gt_ycbcr, 1/scale, 'bicubic');
    im_b_ycbcr = imresize(im_l_ycbcr, scale, 'bicubic');
    im_l_y = im_l_ycbcr(:,:,1) * 255.0;
    im_l = ycbcr2rgb(im_l_ycbcr) * 255.0;
    im_b_y = im_b_ycbcr(:,:,1) * 255.0;
    im_b = ycbcr2rgb(im_b_ycbcr) * 255.0;
    file_no_ext = split(filepaths(i).name, '.');
    file_no_ext = file_no_ext{1};
    filename = sprintf('dataset/mat/%s/%dx/%s.mat', dataset, scale, ...
                       file_no_ext);
    save(filename, 'im_gt_y', 'im_b_y', 'im_gt', 'im_b', 'im_l_ycbcr', 'im_l_y', 'im_l');
end

end