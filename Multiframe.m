%----------Initialization----------
clear all;

% Where to restore result of MAP&Total variation
load_name = '/Users/ganweijie/Downloads/Person-map.mat';
% Where to save result of multi-frame method
save_name = '/Users/ganweijie/Downloads/Person-multiframe.mat';

load(load_name);

[height_lr, width_lr, frames] = size(lr_raw_r);
factor = 4;

height = height_lr * factor;
width = width_lr * factor;

%----------Sparse Phase-------------

out_sparse_r = zeros(height, width, frames);
out_sparse_g = zeros(height, width, frames);
out_sparse_b = zeros(height, width, frames);

A = @(x) idct(x);
AT = @(x) dct(x);

fprintf('Start sparse phase.....\n')
tic
for i = 1: frames
    [theta,~,~,~,~,~]= GPSRBasic(out_map_r(:, :, i),A,2.5,'AT',AT,'Verbose',0);
    out_sparse_r(:,:,i) = A(theta);
    [theta,~,~,~,~,~]= GPSRBasic(out_map_g(:, :, i),A,2.5,'AT',AT,'Verbose',0);
    out_sparse_g(:,:,i) = A(theta);
    [theta,~,~,~,~,~]= GPSRBasic(out_map_b(:, :, i),A,2.5,'AT',AT,'Verbose',0);
    out_sparse_b(:,:,i) = A(theta);

    fprintf('Sparse : No.[%d] frame. The number of total number: [%d] \n', i, frames);
end
sparse_time = toc;

%---------TimeTV Phase--------------

opts.beta  = [0, 0, 1];
opts.print = true;

fprintf('Start sparse phase.....\n')
tic
out_timetv_r = Deconvtvl2(out_sparse_r, 1, 1/0.4, opts);
out_timetv_r = out_timetv_r.f;
out_timetv_g = Deconvtvl2(out_sparse_g, 1, 1/0.4, opts);
out_timetv_g = out_timetv_g.f;
out_timetv_b = Deconvtvl2(out_sparse_b, 1, 1/0.4, opts);
out_timetv_b = out_timetv_b.f;
timetv_time = toc;

%---------Combine channel--------
lr = zeros(height_lr, width_lr, 3, frames);
hr = zeros(height, width, 3, frames);
multi_result = zeros(height, width, 3, frames);

for i = 1 : frames
    lr(:,:,:,i) = cat(3, lr_raw_r(:, :, i), lr_raw_g(:, :, i), lr_raw_b(:, :, i));
    hr(:,:,:,i) = cat(3, hr_raw_r(:, :, i), hr_raw_g(:, :, i), hr_raw_b(:, :, i));
    multi_result(:,:,:,i) = cat(3, out_timetv_r(:, :, i), out_timetv_g(:, :, i), out_timetv_b(:, :, i));
end

% ---------Write Mat file---------------

fprintf('Start writing result to .mat file.....')
lr = uint8(lr);
hr = uint8(hr);
multi_result = uint8(multi_result);

save(save_name, 'lr', 'hr', 'multi_result');