clc
clear
%% Read data
sep15 = single(imread('C:\Users\bnikparv\Downloads\VHRClassified\plt15.tif'));
sep15=sep15(2:end-1,:,:);
sep16 = single(imread('C:\Users\bnikparv\Downloads\planet_order_326105_sep16\clip4_sep_16_2.tif'));
sep16=sep16(2:end-1,:,:);
sep17 = single(imread('C:\Users\bnikparv\Downloads\VHRClassified\plt17.tif'));
sep17=sep17(2:end-1,:,:);
sep18 = single(imread('C:\Users\bnikparv\Downloads\VHRClassified\plt18.tif'));
sep18=sep18(2:end-1,:,:);

lb15 = (imread('C:\Users\bnikparv\Downloads\VHRClassified\rsm_15.tif'));
lb15=lb15(2:end,1:end-1,:);
lb16 = (imread('C:\Users\bnikparv\Downloads\VHRClassified\rsm_16.tif'));
lb16=lb16(2:end,1:end-1,:);
lb17 = (imread('C:\Users\bnikparv\Downloads\VHRClassified\rsm_17.tif'));
lb17=lb17(2:end,1:end-1,:);
lb18 = (imread('C:\Users\bnikparv\Downloads\VHRClassified\rsm_18.tif'));
lb18=lb18(2:end,1:end-1,:);

%% Normalization
m15_1 = max(max(sep15(:,:,1)));
m15_2 = max(max(sep15(:,:,2)));
m15_3 = max(max(sep15(:,:,3)));

m16_1 = max(max(sep16(:,:,1)));
m16_2 = max(max(sep16(:,:,2)));
m16_3 = max(max(sep16(:,:,3)));

m17_1 = max(max(sep17(:,:,1)));
m17_2 = max(max(sep17(:,:,2)));
m17_3 = max(max(sep17(:,:,3)));

m18_1 = max(max(sep18(:,:,1)));
m18_2 = max(max(sep18(:,:,2)));
m18_3 = max(max(sep18(:,:,3)));

m1=max([m16_1, m17_1, m18_1])
m2=max([m16_2, m17_2, m18_2])
m3=max([m16_3, m17_3, m18_3])

sep15n(:,:,1) = sep15(:,:,1)./256;
sep15n(:,:,2) = sep15(:,:,2)./256;
sep15n(:,:,3) = sep15(:,:,3)./256;

sep16n(:,:,1) = sep16(:,:,1)./m16_1;
sep16n(:,:,2) = sep16(:,:,2)./m16_2;
sep16n(:,:,3) = sep16(:,:,3)./m16_3;

sep17n(:,:,1) = sep17(:,:,1)./m17_1;
sep17n(:,:,2) = sep17(:,:,2)./m17_2;
sep17n(:,:,3) = sep17(:,:,3)./m17_3;

sep18n(:,:,1) = sep18(:,:,1)./m18_1;
sep18n(:,:,2) = sep18(:,:,2)./m18_2;
sep18n(:,:,3) = sep18(:,:,3)./m18_3;

all = [sep15n, sep16n, sep17n, sep18n];
all2 = uint16(all.*65535);

alllb = uint8([lb15, lb16, lb17, lb18]);

%%
% imwrite(sep16n, 'sep16n.tif')
% imwrite(sep17n, 'jun17n.tif')
% imwrite( sep18n, 'aug18n.tif')
%%

%% Create test and train data sets
train_data = all2(311:2999, :,:);
train_labels = alllb(311:2999, :,:);
imwrite(train_data, 'train_data.tif');
imwrite(train_labels, 'train_labels.tif');

val_data = all2(3000:end, :,:);
val_labels = alllb(3000:end, :,:);
imwrite(val_data, 'val_data.tif');
imwrite(val_labels, 'val_labels.tif');

%% georeference the images
a = imread('pltall.tif');

%a = imread('lb15gcor0.tif');
% open a file for writing
fid = fopen('all2.txt', 'wt');
% print a title, followed by a blank line
% 
fprintf(fid, 'NCOLS 19800\nNROWS 4235\nXLLCORNER 325329\nYLLCORNER 4018638\n');
fprintf(fid, 'CELLSIZE 3\nNODATA_VALUE 0\n');
% fprintf(fid, 'NCOLS 29630\nNROWS 25353\nXLLCORNER 325355.95\nYLLCORNER 4031346.05\n');
% fprintf(fid, 'CELLSIZE 0.499991422727483\nNODATA_VALUE 0\n');
for ii = 1:size(a,1)
    fprintf(fid,'%g ',a(ii,:));
    fprintf(fid,'\n');
end
fclose(fid);

