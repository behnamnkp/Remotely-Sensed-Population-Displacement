clc;
clear;
%%
% Read all images and sort them by name
%2014
digitDatasetPath14 = fullfile(...
    'D:\Population_displacement\VHRimages\GE_14\train_4class\data');

imds14 = imageDatastore(digitDatasetPath14, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

data14 = {};

for i=1:size(imds14.Files, 1)
   slash_indices = strfind(cell2mat(imds14.Files(i)), '\');
   name = extractAfter(cell2mat(imds14.Files(i)), slash_indices(end));
   if imds14.Labels(i) == 'BG'
       data14(i,1:3)={imds14.Files(i), name, 1};
   elseif imds14.Labels(i) == 'MX'
       data14(i,1:3)={imds14.Files(i), name, 2};
   elseif imds14.Labels(i) == 'RH'
       data14(i,1:3)={imds14.Files(i), name, 3};
   else
       data14(i,1:3)={imds14.Files(i), name, 4};
   end
end

[~, ix] = sort(data14(:,2));
data14(:,:) = data14(ix,:); 

%2015
digitDatasetPath15 = fullfile(...
    'D:\Population_displacement\VHRimages\Pleiade_15_16_18\Train_15\data');

imds15 = imageDatastore(digitDatasetPath15, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

data15 = {};

for i=1:size(imds15.Files, 1)
   slash_indices = strfind(cell2mat(imds15.Files(i)), '\');
   name = extractAfter(cell2mat(imds15.Files(i)), slash_indices(end));
   if imds15.Labels(i) == 'BG'
       data15(i,1:3)={imds15.Files(i), name, 1};
   elseif imds15.Labels(i) == 'MX'
       data15(i,1:3)={imds15.Files(i), name, 2};
   elseif imds15.Labels(i) == 'RH'
       data15(i,1:3)={imds15.Files(i), name, 3};
   else
       data15(i,1:3)={imds15.Files(i), name, 4};
   end
end

[~, ix] = sort(data15(:,2));
data15(:,:) = data15(ix,:); 

%2016
digitDatasetPath16 = fullfile(...
    'D:\Population_displacement\VHRimages\Pleiade_15_16_18\Train_16\data');

imds16 = imageDatastore(digitDatasetPath16, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

data16 = {};

for i=1:size(imds16.Files, 1)
   slash_indices = strfind(cell2mat(imds16.Files(i)), '\');
   name = extractAfter(cell2mat(imds16.Files(i)), slash_indices(end));
   if imds16.Labels(i) == 'BG'
       data16(i,1:3)={imds16.Files(i), name, 1};
   elseif imds16.Labels(i) == 'MX'
       data16(i,1:3)={imds16.Files(i), name, 2};
   elseif imds16.Labels(i) == 'RH'
       data16(i,1:3)={imds16.Files(i), name, 3};
   else
       data16(i,1:3)={imds16.Files(i), name, 4};
   end
end

[~, ix] = sort(data16(:,2));
data16(:,:) = data16(ix,:); 

%2017
digitDatasetPath17 = fullfile(...
    'D:\Population_displacement\VHRimages\WV2_17\Train_17\data');

imds17 = imageDatastore(digitDatasetPath17, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

data17 = {};

for i=1:size(imds17.Files, 1)
   slash_indices = strfind(cell2mat(imds17.Files(i)), '\');
   name = extractAfter(cell2mat(imds17.Files(i)), slash_indices(end));
   if imds17.Labels(i) == 'BG'
       data17(i,1:3)={imds17.Files(i), name, 1};
   elseif imds17.Labels(i) == 'MX'
       data17(i,1:3)={imds17.Files(i), name, 2};
   elseif imds17.Labels(i) == 'RH'
       data17(i,1:3)={imds17.Files(i), name, 3};
   else
       data17(i,1:3)={imds17.Files(i), name, 4};
   end
end

[~, ix] = sort(data17(:,2));
data17(:,:) = data17(ix,:); 

%2018aug
digitDatasetPath18aug = fullfile(...
    'D:\Population_displacement\VHRimages\Pleiade_15_16_18\Train_18_aug\data');

imds18aug = imageDatastore(digitDatasetPath18aug, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

data18aug = {};

for i=1:size(imds18aug.Files, 1)
   slash_indices = strfind(cell2mat(imds18aug.Files(i)), '\');
   name = extractAfter(cell2mat(imds18aug.Files(i)), slash_indices(end));
   if imds18aug.Labels(i) == 'BG'
       data18aug(i,1:3)={imds18aug.Files(i), name, 1};
   elseif imds18aug.Labels(i) == 'MX'
       data18aug(i,1:3)={imds18aug.Files(i), name, 2};
   elseif imds18aug.Labels(i) == 'RH'
       data18aug(i,1:3)={imds18aug.Files(i), name, 3};
   else
       data18aug(i,1:3)={imds18aug.Files(i), name, 4};
   end
end

[~, ix] = sort(data18aug(:,2));
data18aug(:,:) = data18aug(ix,:); 
%% 
% Create a cell of all years
d(:,1:3) = data14;
d(:,4:6) = data15;
d(:,7:9) = data16;
d(:,10:12) = data17;
d(:,13:15) = data18aug;

save('trainTraj', 'd');

%%
% Create a GUI to represent and change the values of the matrix

%%
% save images into their new label folders
load('trainTraj_4643.mat')
for i=1:size(d, 1)
   im_ref = d(i,1:3);
   image = imread(string(im_ref(1)));
   if cell2mat(im_ref(3))==1
       name = 'G:\backupC27152020\Population_Displacement_Final\Resources\VHR\correctedTraining\BG\' + string(im_ref(2))
       imwrite(image, name);
   elseif cell2mat(im_ref(3))==2
       name = 'G:\backupC27152020\Population_Displacement_Final\Resources\VHR\correctedTraining\MX\' + string(im_ref(2))
       imwrite(image, name);
   elseif cell2mat(im_ref(3))==3
       name = 'G:\backupC27152020\Population_Displacement_Final\Resources\VHR\correctedTraining\RH\' + string(im_ref(2))
       imwrite(image, name);
   else
       name = 'G:\backupC27152020\Population_Displacement_Final\Resources\VHR\correctedTraining\RN\' + string(im_ref(2))
       imwrite(image, name);
   end
end
%%
%trajectory analysis
% training trajectories
load('trainTraj_4643.mat')
%labels_train=cell2mat(d(:,[3 6 9 12 15]));

traintrajchange(:,1) = cell2mat(d(:,6))-cell2mat(d(:,3));
traintrajchange(:,2) = cell2mat(d(:,9))-cell2mat(d(:,6));
traintrajchange(:,3) = cell2mat(d(:,12))-cell2mat(d(:,9));
traintrajchange(:,4) = cell2mat(d(:,15))-cell2mat(d(:,12));
traintrajchange(traintrajchange~=0)=1;

x=sum(traintrajchange,2);
histogram(x, 'BinMethod','integers','orientation', 'horizontal')
yticks([0, 1, 2, 3, 4])

x2 = ceil((histcounts(x)/sum(histcounts(x))*100));
bar(x2);
text(1:length(x2),x2,num2str(x2'),'vert','bottom','horiz','center'); 
box off

n=1;
for i=1:size(traintrajchange, 1)
    if sum(traintrajchange(i, :))>2
        d2(n,:) = d(i,:);
        n=n+1;
    else     
    end
end
labels_train = string(cell2mat(d2(:,3))) + string(cell2mat(d2(:,6))) + ...
    string(cell2mat(d2(:,9))) + string(cell2mat(d2(:,12))) + ...
    string(cell2mat(d2(:,15)));

for i=1:size(labels_train, 1)
    lb_train(i, 1) = str2num(labels_train(i));
end

unq_train_traj = unique(lb_train,'rows');
[N,edges,bin]=histcounts(lb_train,'BinMethod','integers');
q=1;
for i=1:size(N,2)
    if N(i)~=0
       M(q) = N(i);
       q=q+1;
    end
end

% classified images trajectories
imagetrajchange(:,1) = trajectories(:,2)-trajectories(:,1);
imagetrajchange(:,2) = trajectories(:,3)-trajectories(:,2);
imagetrajchange(:,3) = trajectories(:,4)-trajectories(:,3);
imagetrajchange(:,4) = trajectories(:,5)-trajectories(:,4);
imagetrajchange(imagetrajchange~=0)=1;

xx=sum(imagetrajchange,2);
histogram(xx, 'BinMethod','integers','orientation', 'horizontal')
yticks([0, 1, 2, 3, 4])

xx2 = ceil((histcounts(xx)/sum(histcounts(xx))*100));
bar(xx2);
text(1:length(xx2),xx2,num2str(xx2'),'vert','bottom','horiz','center'); 
box off

m=1;
for i=1:size(imagetrajchange, 1)
    if sum(imagetrajchange(i, :))>1
        trajectories2(i,:) = trajectories(i,:);
        %m=m+1;
    else
        trajectories2(i,:) = [-999 -999 -999 -999 -999];
    end
end

labels_image = string(trajectories2(:,1)) + string(trajectories2(:,2)) + ...
    string(trajectories2(:,3)) + string(trajectories2(:,4)) + ...
    string(trajectories2(:,5));

for i=1:size(labels_image, 1)
    lb_image(i, 1) = str2num(labels_image(i));
end

unq_image_traj = unique(lb_image,'rows');