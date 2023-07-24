clc;
clear;
%% Read train data
digitDatasetPath = fullfile(...
    'D:\Population_displacement\VHRimages\traincorrections\2014\data');

imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

[~,sample] = splitEachLabel(imds,0.8);

[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7);

%%
labelCountTrain = countEachLabel(imdsTrain)
labelCountTvalidation = countEachLabel(imdsValidation)
%%
img = readimage(imdsTrain,1);
size(img)
% %% 
% inputBatch = preview(imdsTrain);
% disp(inputBatch)
% 
% inputTileSize = [256,256,3];
% lgraph = createUnet(inputTileSize);
% disp(lgraph.Layers)
% 
% newClassLayer = classificationLayer('Name','classoutput');
% lgraph = replaceLayer(lgraph,lgraph.Layers(58).Name,newClassLayer);
% analyzeNetwork(lgraph)

%% GoogLenet 
net = googlenet;
%net.Layers
inputSize = net.Layers(1).InputSize
%analyzeNetwork(net)

numClasses = numel(categories(imdsTrain.Labels))
%layers(1) = imageInputLayer([50 50 3], 'Name', 'data');

%% 
net = xception;

inputSize = [299 299];
imdsTrain.ReadFcn = @(loc)imresize(imread(loc),inputSize);
imdsValidation.ReadFcn = @(loc)imresize(imread(loc),inputSize);

%inputSize = net.Layers(1).InputSize
%analyzeNetwork(net)

numClasses = numel(categories(imdsTrain.Labels))
%layers(1) = imageInputLayer([50 50 3], 'Name', 'data');

%%
if isa(net,'SeriesNetwork')
    lgraph = layerGraph(net.Layers);
else
    lgraph = layerGraph(net);
end

[learnableLayer,classLayer] = findLayersToReplace(lgraph);
[learnableLayer,classLayer]

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnablefindLayersLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);
% 
% figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
% plot(lgraph)
% ylim([0,10])

%%
  pixelRange = [-30 30];
  imageAugmenter = imageDataAugmenter( ...
      'RandXReflection',true, ...
      'RandXTranslation',pixelRange, ...
       'RandYTranslation',pixelRange);
% augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
%     'DataAugmentation',imageAugmenter);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

options = trainingOptions('adam', ...
    'MiniBatchSize',32, ...
    'MaxEpochs',15, ...
    'InitialLearnRate',1e-3, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',true, ...
    'Plots','training-progress');
%%
netTransfer = trainNetwork(augimdsTrain,lgraph,options);
%filename = 'D:\Population_displacement\VHR images\GE_14\net_GE_14.onnx';
%exportONNXNetwork(netTransfer,filename)

%%
modelfile = 'D:\Population_displacement\VHRimages\traincorrections\2014\net_14_8_5_20.onnx';
classes = ["BG","MX","RH", "RN"];
netTransfer = importONNXNetwork(modelfile,'OutputLayerType','classification','Classes',classes)

tic
YPred = classify(netTransfer,augimdsValidation);
toc
YValidation = imdsValidation.Labels;

figure('Units','normalized','Position',[0.2 0.2 0.4 0.4]);
cm = confusionchart(YValidation,YPred);
cm.Title = 'Confusion Matrix for Validation Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

accuracy = sum(YPred == YValidation)/numel(YValidation)

%%
clc;
clear;
modelfile = 'D:\Population_displacement\VHRimages\traincorrections\2014\net_14.onnx';
classes = ["BG" "MX" "RH" "RN"];
net = importONNXNetwork(modelfile,'OutputLayerType','classification','Classes',classes)

%%
img_ = imread('D:\Population_displacement\VHR images\Pleiade_15_16_18\Pansharpening_aug_16_clip.tif');
%img_ = imread('D:\Population_displacement\VHR images\Pleiade_15_16_18\Pansharpening_nov_15_clip.tif');
%img_ = imread('D:\Population_displacement\VHR images\Pleiade_15_16_18\Pansharpening_aug_16.tif');
%img_ = imread('D:\Population_displacement\VHR images\WV2_17\Pansharpening_June_17.tif');
%img_ = imread('D:\Population_displacement\VHR images\Pleiade_15_16_18\Pansharpening_may18.tif');
%img_ = imread('D:\Population_displacement\VHR images\Pleiade_15_16_18\Pansharpening_aug_18.tif');
img_ = img_(:,:,1:3);

size(img_)
windowSize = 100

%ge_aug_2014
%nrow=26301
%ncol=37401
%pl_nov_15
nrow=25301
ncol=29601
%pl_aug_16
%nrow=31301
%ncol=32801
%vw_jun_17
%nrow=28901
%ncol=30901
%pl_may_18
%nrow=31301
%ncol=32901
%pl_aug_2018
%nrow=28701
%ncol=39401
label_ = zeros([nrow ncol 4]);
score_ = zeros([nrow ncol 4]);
bgCount = 0;
rhCount = 0;
mxCount = 0;

bgCount2 = 0;
rhCount2 = 0;
mxCount2 = 0;

for i=1:windowSize:nrow
   for j=1:windowSize:ncol 
       window = img_(i:i+windowSize-1, j:j+windowSize-1, :);
       window  = single(window(:,:,1:3));
       window_rs = imresize(window, [224 224]);
       
       max_ = max(max(max(window_rs)));
       min_ = min(min(min(window_rs)));

       window_norm = (window_rs - min_) / (max_ - min_);
       max_n = max(max(max(window_norm)));
       min_n = min(min(min(window_norm)));

       window2 = uint16(window_norm * 65536);
    
       [label,scores] = classify(net,window2);
       score_(i:i+windowSize-1, j:j+windowSize-1, 1) = scores(1);
       score_(i:i+windowSize-1, j:j+windowSize-1, 2) = scores(2);
       score_(i:i+windowSize-1, j:j+windowSize-1, 3) = scores(3);
       score_(i:i+windowSize-1, j:j+windowSize-1, 4) = scores(4);
       
       [~,idx] = sort(scores,'descend');
       idx = idx(1:2);
       classNamesTop = net.Layers(end).ClassNames(idx);
       scoresTop = scores(idx);
       
       if label == 'BG'
           label_(i:i+windowSize-1, j:j+windowSize-1, 1) = 1;
           bgCount = bgCount + 1;
       elseif label == 'MX'
           label_(i:i+windowSize-1, j:j+windowSize-1, 1) = 2;
           mxCount = mxCount + 1;
       else
           label_(i:i+windowSize-1, j:j+windowSize-1, 1) = 3;
           rhCount = rhCount + 1;
       end
       
       label_(i:i+windowSize-1, j:j+windowSize-1, 2) = scoresTop(1);
       
       if string(classNamesTop{2,1}) == 'BG'
           label_(i:i+windowSize-1, j:j+windowSize-1, 3) = 1;
           bgCount2 = bgCount2 + 1;
       elseif string(classNamesTop{2,1}) == 'MX'
           label_(i:i+windowSize-1, j:j+windowSize-1, 3) = 2;
           mxCount2 = mxCount2 + 1;
       else
           label_(i:i+windowSize-1, j:j+windowSize-1, 3) = 3;
           mxCount2 = mxCount2 + 1;
       end
       
       label_(i:i+windowSize-1, j:j+windowSize-1, 4) = scoresTop(2);
       
       label;
   end
   i
end
%imshow(label_(:, :, 1))
imwrite(uint8(label_(:,:,1)), 'label_all_15.tif');
imwrite(uint8(score_), 'score_all_15.tif');
%class = [bgCount mxCount rhCount];
%save('class_aug_18.mat');
%class2 = [bgCount2 mxCount2 rhCount2];
%save('class2_aug_18.mat');

%% consistency evaluation
clc;
clear;
lb14g = imread('D:\Population_displacement\VHRimages\traincorrections\2014\label_14_2.tif');
lb15g = imread('D:\Population_displacement\VHRimages\traincorrections\2015\label_15.tif');
lb16g = imread('D:\Population_displacement\VHRimages\traincorrections\2016\label_16.tif');
lb17g = imread('D:\Population_displacement\VHRimages\traincorrections\2017\label_17.tif');
lb18augg = imread('D:\Population_displacement\VHRimages\traincorrections\2018_AUG\label_18aug.tif');

all(:,:,1)= lb14g;
all(:,:,2)= lb15g;
all(:,:,3)= lb16g;
all(:,:,4)= lb17g;
all(:,:,5)= lb18augg;

allpatches=zeros(254, 297, 5);
patchsize=100
for k=1:size(all, 3)
    for i=1:patchsize:size(all, 1)
        for j=1:patchsize:size(all, 2)
            allpatches(((i-1)/patchsize)+1, ... 
                ((j-1)/patchsize)+1, k) = ...
                mode(mode(all(i:i+patchsize-1, j:j+patchsize-1,k)));
        end
    end
end
%%
trajectories = reshape(allpatches, [254*297, 5]);
trajchange(:,1) = trajectories(:,2)-trajectories(:,1);
trajchange(:,2) = trajectories(:,3)-trajectories(:,2);
trajchange(:,3) = trajectories(:,4)-trajectories(:,3);
trajchange(:,4) = trajectories(:,5)-trajectories(:,4);
trajchange(trajchange~=0)=1;

chng = sum(trajchange, 2);
unqclssnum = zeros(75438, 1);
for i=1:size(trajectories, 1)
   unqclssnum(i, 1) = size(unique(trajectories(i,:)), 2);
end

unqclss = zeros(75438, 1);
for i=1:size(trajectories, 1)
   aux = unique(trajectories(i,:));
   j=1;
   ss = '';
   while j<=size(aux, 2)
       ss = ss + string(aux(1,j));
       j=j+1;
   end
   unqclss(i, 1) = ss;
end

% trajectories2 = zeros(size(trajectories));
% for i=1:size(trajchange, 1)
%    if sum(trajchange(i, :)) > 1
%        trajectories2(i,:) = mode(trajectories(i,:));
%    else
%        trajectories2(i,:) = trajectories(i,:);
%    end
% end
% 
% trajectories3 = reshape(trajectories2, [254, 297, 5]);

modes=zeros(size(trajectories));
for i=1:size(trajectories, 1)
   if sum(trajchange(i,:))>=0
       for j=1:size(trajectories, 2)
           switch j
           case 1
              idx_bg = find(trajectories(i,:)==1);
              bg = 0.438*(sum(1./idx_bg));
              idx_mx = find(trajectories(i,:)==2);
              mx = 0.438*(sum(1./idx_mx));
              idx_rh = find(trajectories(i,:)==3);
              rh = 0.438*(sum(1./idx_rh));
              idx_rn = find(trajectories(i,:)==4);
              rn = 0.438*(sum(1./idx_rn));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx;
           case 2
              dd = [1 0 1 2 3];
              idx_bg = find(trajectories(i,:)==1);
              bg = 0.438*(sum(1./(dd(idx_bg)+1)));
              idx_mx = find(trajectories(i,:)==2);
              mx = 0.438*(sum(1./(dd(idx_mx)+1)));
              idx_rh = find(trajectories(i,:)==3);
              rh = 0.438*(sum(1./(dd(idx_rh)+1)));
              idx_rn = find(trajectories(i,:)==4);
              rn = 0.438*(sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx;
           case 3
              dd = [2 1 0 1 2];
              idx_bg = find(trajectories(i,:)==1);
              bg = 0.438*(sum(1./(dd(idx_bg)+1)));
              idx_mx = find(trajectories(i,:)==2);
              mx = 0.438*(sum(1./(dd(idx_mx)+1)));
              idx_rh = find(trajectories(i,:)==3);
              rh = 0.438*(sum(1./(dd(idx_rh)+1)));
              idx_rn = find(trajectories(i,:)==4);
              rn = 0.438*(sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx;
           case 4
              dd = [3 2 1 0 1];
              idx_bg = find(trajectories(i,:)==1);
              bg = 0.438*(sum(1./(dd(idx_bg)+1)));
              idx_mx = find(trajectories(i,:)==2);
              mx = 0.438*(sum(1./(dd(idx_mx)+1)));
              idx_rh = find(trajectories(i,:)==3);
              rh = 0.438*(sum(1./(dd(idx_rh)+1)));
              idx_rn = find(trajectories(i,:)==4);
              rn = 0.438*(sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx;
           otherwise
              dd = [4 3 2 1 0];
              idx_bg = find(trajectories(i,:)==1);
              bg = 0.438*(sum(1./(dd(idx_bg)+1)));
              idx_mx = find(trajectories(i,:)==2);
              mx = 0.438*(sum(1./(dd(idx_mx)+1)));
              idx_rh = find(trajectories(i,:)==3);
              rh = 0.438*(sum(1./(dd(idx_rh)+1)));
              idx_rn = find(trajectories(i,:)==4);
              rn = 0.438*(sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx;
           end
       end
   else
       modes(i,:)=trajectories(i,:);
   end
end

modes2=reshape(modes, [254 297 5]);
%%
% modes = zeros(size(trajectories, 1), 8);
% for i=1:size(modes, 1)
%     modes(i, 1:8) = [1 sum(trajectories(i,:)==1) 2 sum(trajectories(i,:)==2) ...
%         3 sum(trajectories(i,:)==3) 4 sum(trajectories(i,:)==4)];
% end

spatialmode = zeros(size(allpatches, 1)*size(allpatches, 2)*size(allpatches, 3)...
    , 8);
ker_r = 2;
m=1;
for k=1:size(allpatches, 3)
    for j=3:size(allpatches, 2)-2
       for i=3:size(allpatches, 1)-2
           if (i==1) && (j==1)
              f = allpatches(i:i+ker_r, j:j+ker_r,k);
              filter=reshape(f, [1 ((ker_r+1)^2)]);
              spatialmode(m, 1:8) = [1 sum(filter==1)/((ker_r+1)^2) 2 sum(filter==2)/((ker_r+1)^2) ...
              3 sum(filter==3)/((ker_r+1)^2) 4 sum(filter==4)/((ker_r+1)^2)];
              m=m+1;
           elseif i==1 && j==size(allpatches, 2)
              f = allpatches(i:i+ker_r, j-ker_r:j,k);
              filter=reshape(f, [1 ((ker_r+1)^2)]);
              spatialmode(m, 1:8) = [1 sum(filter==1)/((ker_r+1)^2) 2 sum(filter==2)/((ker_r+1)^2) ...
              3 sum(filter==3)/((ker_r+1)^2) 4 sum(filter==4)/((ker_r+1)^2)];
              m=m+1;
           elseif i==size(allpatches, 1) && j==size(allpatches, 2)
              f = allpatches(i-ker_r:i, j-ker_r:j,k);
              filter=reshape(f, [1 ((ker_r+1)^2)]);
              spatialmode(m, 1:8) = [1 sum(filter==1)/((ker_r+1)^2) 2 sum(filter==2)/((ker_r+1)^2) ...
              3 sum(filter==3)/((ker_r+1)^2) 4 sum(filter==4)/((ker_r+1)^2)];
              m=m+1;
           elseif i==size(allpatches, 1) && j==1
              f = allpatches(i-ker_r:i, j:j+ker_r,k);
              filter=reshape(f, [1 ((ker_r+1)^2)]);
              spatialmode(m, 1:8) = [1 sum(filter==1)/((ker_r+1)^2) 2 sum(filter==2)/((ker_r+1)^2) ...
              3 sum(filter==3)/((ker_r+1)^2) 4 sum(filter==4)/((ker_r+1)^2)];
              m=m+1;
           elseif i==1 && j~=1 && j~=size(allpatches, 2)
              f = allpatches(i:i+ker_r, j-ker_r:j+ker_r,k);
              filter=reshape(f, [1 ((ker_r+1)*(2*ker_r+1))]);
              spatialmode(m, 1:8) = [1 sum(filter==1)/((ker_r+1)*(2*ker_r+1)) 2 sum(filter==2)/((ker_r+1)*(2*ker_r+1)) ...
              3 sum(filter==3)/((ker_r+1)*(2*ker_r+1)) 4 sum(filter==4)/((ker_r+1)*(2*ker_r+1))];
              m=m+1;
           elseif i~=1 && i~=size(allpatches, 1) && j==size(allpatches, 2)
              f = allpatches(i-ker_r:i+ker_r, j-ker_r:j,k);
              filter=reshape(f, [1 ((ker_r+1)*(2*ker_r+1))]);
              spatialmode(m, 1:8) = [1 sum(filter==1)/((ker_r+1)*(2*ker_r+1)) 2 sum(filter==2)/((ker_r+1)*(2*ker_r+1)) ...
              3 sum(filter==3)/((ker_r+1)*(2*ker_r+1)) 4 sum(filter==4)/((ker_r+1)*(2*ker_r+1))];
              m=m+1;
           elseif i==size(allpatches, 1) && j~=size(allpatches, 2) && j~=1
              f = allpatches(i-ker_r:i, j-ker_r:j+ker_r,k);
              filter=reshape(f, [1 ((ker_r+1)*(2*ker_r+1))]);
              spatialmode(m, 1:8) = [1 sum(filter==1)/((ker_r+1)*(2*ker_r+1)) 2 sum(filter==2)/((ker_r+1)*(2*ker_r+1)) ...
              3 sum(filter==3)/((ker_r+1)*(2*ker_r+1)) 4 sum(filter==4)/((ker_r+1)*(2*ker_r+1))];
              m=m+1;
           elseif i~=1 && i~=size(allpatches, 1) && j==1
              f = allpatches(i-ker_r:i+ker_r, j:j+ker_r,k);
              filter=reshape(f, [1 ((ker_r+1)*(2*ker_r+1))]);
              spatialmode(m, 1:8) = [1 sum(filter==1)/((ker_r+1)*(2*ker_r+1)) 2 sum(filter==2)/((ker_r+1)*(2*ker_r+1)) ...
              3 sum(filter==3)/((ker_r+1)*(2*ker_r+1)) 4 sum(filter==4)/((ker_r+1)*(2*ker_r+1))];
              m=m+1;
           else
               f = allpatches(i-ker_r:i+ker_r, j-ker_r:j+ker_r,k);
               filter=reshape(f, [1 ((2*ker_r+1)^2)]);
               spatialmode(m, 1:8) = [1 sum(filter==1)/((2*ker_r+1)^2) 2 sum(filter==2)/((2*ker_r+1)^2) ...
                   3 sum(filter==3)/((2*ker_r+1)^2) 4 sum(filter==4)/((2*ker_r+1)^2)];
               m=m+1;
           end
       end
    end
end

spatialmode2=reshape(spatialmode, [297*254 5 8]);

corrections = zeros(size(spatialmode2, 1), 5);
for i=1:size(spatialmode2, 1)
   for j=1:size(spatialmode2,2)
       bg_t = modes(i, j, 2);
       mx_t = modes(i, j, 4);
       rn_t = modes(i, j, 6);
       in_t = modes(i, j, 8);
       
       bg_s = spatialmode2(i, j, 2);
       mx_s = spatialmode2(i, j, 4);
       rn_s = spatialmode2(i, j, 6);
       in_s = spatialmode2(i, j, 8);
       tw = 1;
       sw = 0;
       
       scores = [bg_t*tw + bg_s*sw mx_t*tw + mx_s*sw rn_t*tw + rn_s*sw in_t*tw + in_s*sw];
       [M,I] = max(scores);
       
       if sum(trajchange(i, :))>1
           corrections(i,j) = I;
       else
           %corrections(i,j) = 0;
           corrections(i,j) = trajectories(i, j);
       end
   end
end

corrections2=reshape(corrections, [254 297 5]);

%% reverse size
clear
clc
lb15g = imread('label_15g_f_3.tif');
reverse = zeros(25400, 29700, 1);
patchsize=100;
for k=1:size(lb15g, 3)
    for i=1:size(lb15g, 1)
        for j=1:size(lb15g, 2)
            reverse((i-1)*patchsize+1:i*patchsize,...
                (j-1)*patchsize+1:j*patchsize,k)=...
                lb15g(i,j,k);
        end
    end
end
%%
%imwrite(uint8(reverse), 'label_14g_M_2cc.tif');

% georeference images
a = uint8(lb14g);
%a = imread('lb15gcor0.tif');
% open a file for writing
fid = fopen('lb14_2g.txt', 'wt');
% print a title, followed by a blank line
% fprintf(fid, 'NCOLS 29700\nNROWS 25400\nXLLCORNER 325329.094284\nYLLCORNER 4018640.266483\n');
% fprintf(fid, 'CELLSIZE 0.499991422727483\nNODATA_VALUE 0\n');
fprintf(fid, 'NCOLS 29700\nNROWS 25400\nXLLCORNER 325355.95\nYLLCORNER 4031346.05\n');
fprintf(fid, 'CELLSIZE 0.499991422727483\nNODATA_VALUE 0\n');
for ii = 1:size(a,1)
    fprintf(fid,'%g ',a(ii,:));
    fprintf(fid,'\n');
end
fclose(fid);

% figure;
% imagesc(uint8(all(:,:,3)));
% classNames=[ "BG";"MX";"RH";"IN"];
% ticks = 1/(8*2):1/8:1;
% colorbar('TickLabels',cellstr(classNames),'Ticks',ticks,...
%     'TickLength',0,'TickLabelInterpreter','none')
% colormap jet(4)
% 
% figure;
% imagesc(uint8(trajectories3(:,:,3)));
% classNames=[ "BG";"MX";"RH";"IN"];
% ticks = 1/(8*2):1/8:1;
% colorbar('TickLabels',cellstr(classNames),'Ticks',ticks,...
%     'TickLength',0,'TickLabelInterpreter','none')
% colormap jet(4)
%% training data check for correction
% temporal
clc;
clear;
load('trainTraj_4643.mat');

traincheck(:,1) = d(:,3);
traincheck(:,2) = d(:,6);
traincheck(:,3) = d(:,9);
traincheck(:,4) = d(:,12);
traincheck(:,5) = d(:,15);

traincheck = cell2mat(traincheck);

trajchange(:,1) = traincheck(:,2)-traincheck(:,1);
trajchange(:,2) = traincheck(:,3)-traincheck(:,2);
trajchange(:,3) = traincheck(:,4)-traincheck(:,3);
trajchange(:,4) = traincheck(:,5)-traincheck(:,4);
trajchange(trajchange~=0)=1;

modes=zeros(size(traincheck, 1), 8);
for i=1:size(traincheck, 1)
   if sum(trajchange(i))>1
       for j=1:size(traincheck, 2)
           switch j
           case 1
              idx_bg = find(traincheck(i,:)==1);
              bg = 0.438*(sum(1./idx_bg));
              idx_mx = find(traincheck(i,:)==2);
              mx = 0.438*(sum(1./idx_mx));
              idx_rh = find(traincheck(i,:)==3);
              rh = 0.438*(sum(1./idx_rh));
              idx_rn = find(traincheck(i,:)==4);
              rn = 0.438*(sum(1./idx_rn));
              [~, idx] = max([bg mx rh rn]);
              modes(i,j)=idx;
           case 2
              dd = [1 0 1 2 3];
              idx_bg = find(traincheck(i,:)==1);
              bg = 0.438*(sum(1./(dd(idx_bg)+1)));
              idx_mx = find(traincheck(i,:)==2);
              mx = 0.438*(sum(1./(dd(idx_mx)+1)));
              idx_rh = find(traincheck(i,:)==3);
              rh = 0.438*(sum(1./(dd(idx_rh)+1)));
              idx_rn = find(traincheck(i,:)==4);
              rn = 0.438*(sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);;
              modes(i,j)=idx;
           case 3
              dd = [2 1 0 1 2];
              idx_bg = find(traincheck(i,:)==1);
              bg = 0.438*(sum(1./(dd(idx_bg)+1)));
              idx_mx = find(traincheck(i,:)==2);
              mx = 0.438*(sum(1./(dd(idx_mx)+1)));
              idx_rh = find(traincheck(i,:)==3);
              rh = 0.438*(sum(1./(dd(idx_rh)+1)));
              idx_rn = find(traincheck(i,:)==4);
              rn = 0.438*(sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              modes(i,j)=idx;
           case 4
              dd = [3 2 1 0 1];
              idx_bg = find(traincheck(i,:)==1);
              bg = 0.438*(sum(1./(dd(idx_bg)+1)));
              idx_mx = find(traincheck(i,:)==2);
              mx = 0.438*(sum(1./(dd(idx_mx)+1)));
              idx_rh = find(traincheck(i,:)==3);
              rh = 0.438*(sum(1./(dd(idx_rh)+1)));
              idx_rn = find(traincheck(i,:)==4);
              rn = 0.438*(sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              modes(i,j)=idx;
           otherwise
              dd = [4 3 2 1 0];
              idx_bg = find(traincheck(i,:)==1);
              bg = 0.438*(sum(1./(dd(idx_bg)+1)));
              idx_mx = find(traincheck(i,:)==2);
              mx = 0.438*(sum(1./(dd(idx_mx)+1)));
              idx_rh = find(traincheck(i,:)==3);
              rh = 0.438*(sum(1./(dd(idx_rh)+1)));
              idx_rn = find(traincheck(i,:)==4);
              rn = 0.438*(sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              modes(i,j)=idx;
           end
       end
   else
       modes(i,:)=(traincheck(i,:));
   end
end

% spatial 

all = imread('all.tif');
all = all(all ~= 0);

label2_14g_all = imread('label2_14g_all.tif');
label2_14g_all = label2_14g_all(label2_14g_all ~= 15);
lb14_1upc_all = imread('lb14_1upc_all.tif');
lb14_1upc_all = lb14_1upc_all(lb14_1upc_all ~= 15);
label_14g_M_2ccg_all = imread('label_14g_M_2ccg_all.tif');
label_14g_M_2ccg_all = label_14g_M_2ccg_all(label_14g_M_2ccg_all ~= 15);
label_14g_M_3ccg_all = imread('label_14g_M_3ccg_all.tif');
label_14g_M_3ccg_all = label_14g_M_3ccg_all(label_14g_M_3ccg_all ~= 15);
label_14g_M_4ccg_all = imread('label_14g_M_4ccg_all.tif');
label_14g_M_4ccg_all = label_14g_M_4ccg_all(label_14g_M_4ccg_all ~= 15);

YPred = label_14g_M_4ccg_all;
YValidation = all;

figure('Units','normalized','Position',[0.2 0.2 0.4 0.4]);
cm = confusionchart(YValidation,YPred);
cm.Title = 'Confusion Matrix for Validation Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

accuracy = sum(YPred == YValidation)/numel(YValidation)
%% traning geographic corrections
clc;
clear;
load('G:\backupC27152020\C\models2\matlab\trainTraj_4643.mat');

listing = dir(...
    'C:\Users\bnikparv\Downloads\DATASETS\train\17');

uniquenames = {};
for i=3:size(listing, 1)
    baseFileName = listing(i).name;
    if contains(baseFileName, '.xml') || contains(baseFileName, '.ovr') ...
            || contains(baseFileName, 'tfw') || contains(baseFileName, '.aux') ...
            || contains(baseFileName, '.TIF')
       baseFileName2 = replace(baseFileName,'.xml','');
       baseFileName2 = replace(baseFileName2,'.ovr','');
       baseFileName2 = replace(baseFileName2,'.tfw','');
       baseFileName2 = replace(baseFileName2,'.aux','');
       baseFileName2 = replace(baseFileName2,'.TIF','');
    else
    end 
    idx = strfind(uniquenames, baseFileName2);
    idx2 = 0;
    for j=1:size(idx,2)
        if cell2mat(idx(j)) == 1
            idx2 = j;
        else
        end
    end
    if idx2==0
        uniquenames{end+1} = baseFileName2;
    else
    end
end
uniquenames = uniquenames';
      
for i=3:size(listing, 1)
    baseFileName = listing(i).name;
    if contains(baseFileName, '.xml') || contains(baseFileName, '.ovr') ...
            || contains(baseFileName, 'tfw') || contains(baseFileName, '.aux') ...
            || contains(baseFileName, '.TIF')
       baseFileName2 = replace(baseFileName,'.xml','');
       baseFileName2 = replace(baseFileName2,'.ovr','');
       baseFileName2 = replace(baseFileName2,'.tfw','');
       baseFileName2 = replace(baseFileName2,'.aux','');
       baseFileName2 = replace(baseFileName2,'.TIF','');
    else
    end 
    
    idx = strfind(uniquenames, baseFileName2);
    idx2=0;
    for j=1:size(idx,1)
       if cell2mat(idx(j)) == 1
           idx2 = j;
       else
       end
    end
    
    idx_ = strfind(string(d(:, 11)), string(uniquenames(idx2)) + '.TIF');
    idx3=0;
    for j=1:size(idx_,1)
       if cell2mat(idx_(j)) == 1
           idx3 = j;
       else
       end
    end
    label = cell2mat(d(idx3,12));
    
    switch label 
        case 1
            n1= strcat('bg17_' + string(idx2));
            n2 = replace(baseFileName, baseFileName2, n1);
            movefile(listing(i).name, n2);

        case 2
            n1= strcat('mx17_' + string(idx2));
            n2 = replace(baseFileName, baseFileName2, n1);
            movefile(listing(i).name, n2);

        case 3
            n1= strcat('rh17_' + string(idx2));
            n2 = replace(baseFileName, baseFileName2, n1);
            movefile(listing(i).name, n2);
            
        otherwise
            n1= strcat('rn17_' + string(idx2));
            n2 = replace(baseFileName, baseFileName2, n1);
            movefile(listing(i).name, n2);
    end
end

%%
% trajectory check 

lb(:,:,1)=imread('D:\Population_displacement\VHRimages\GE_14\train_4class\lb14gcor0.tif');
lb(:,:,2)=imread('D:\Population_displacement\VHRimages\GE_14\train_4class\lb14gcor10.tif');
lb(:,:,3)=imread('D:\Population_displacement\VHRimages\GE_14\train_4class\lb14gcor20.tif');
lb(:,:,4)=imread('D:\Population_displacement\VHRimages\GE_14\train_4class\lb14gcor30.tif');
lb(:,:,5)=imread('D:\Population_displacement\VHRimages\GE_14\train_4class\lb14gcor40.tif');
lb(:,:,6)=imread('D:\Population_displacement\VHRimages\GE_14\train_4class\lb14gcor50.tif');
lb(:,:,7)=imread('D:\Population_displacement\VHRimages\GE_14\train_4class\lb14gcor60.tif');
lb(:,:,8)=imread('D:\Population_displacement\VHRimages\GE_14\train_4class\lb14gcor70.tif');
lb(:,:,9)=imread('D:\Population_displacement\VHRimages\GE_14\train_4class\lb14gcor80.tif');
lb(:,:,10)=imread('D:\Population_displacement\VHRimages\GE_14\train_4class\lb14gcor90.tif');
lb(:,:,11)=imread('D:\Population_displacement\VHRimages\GE_14\train_4class\lb14gcor100.tif');
cnts(1,:) = (histcounts(lb(:,:,1))/sum(histcounts(lb(:,:,1))))*100;
cnts(2,:) = (histcounts(lb(:,:,2))/sum(histcounts(lb(:,:,2))))*100;
cnts(3,:) = (histcounts(lb(:,:,3))/sum(histcounts(lb(:,:,3))))*100;
cnts(4,:) = (histcounts(lb(:,:,4))/sum(histcounts(lb(:,:,4))))*100;
cnts(5,:) = (histcounts(lb(:,:,5))/sum(histcounts(lb(:,:,5))))*100;
cnts(6,:) = (histcounts(lb(:,:,6))/sum(histcounts(lb(:,:,6))))*100;
cnts(7,:) = (histcounts(lb(:,:,7))/sum(histcounts(lb(:,:,7))))*100;
cnts(8,:) = (histcounts(lb(:,:,8))/sum(histcounts(lb(:,:,8))))*100;
cnts(9,:) = (histcounts(lb(:,:,9))/sum(histcounts(lb(:,:,9))))*100;
cnts(10,:) = (histcounts(lb(:,:,10))/sum(histcounts(lb(:,:,10))))*100;
cnts(11,:) = (histcounts(lb(:,:,11))/sum(histcounts(lb(:,:,11))))*100;

h=figure;
hold on
for ii = 1:4
 plot(cnts(:,ii))
 legend(['BG'; 'MX'; 'RH'; 'IN'],'Location','NorthEastOutside')
end
title('2014')
ylabel('class proportion (percentage)')
xlabel('time-space percentage')
%xticklabels({'0','10','20','30','40','50','60','70','80','90','100'})
grid on;
datacursormode(h,'on');
set(gcf, 'Position',  [100, 100, 400, 600])
name = 'WEIGHT2014.png';
saveas(gcf,name);

% figure;
% C = [0 2 4 6; 8 10 12 14; 16 18 20 22];
% for f = 1:size(lb, 3)
%     I=imagesc(uint8(lb(:,:,f)));
%     classNames=[ "BG";"MX";"RH";"IN"];
%     ticks = 1/(8*2):1/8:1;
%     colorbar('TickLabels',cellstr(classNames),'Ticks',ticks,...
%     'TickLength',0,'TickLabelInterpreter','none')
%     colormap jet(4)
%     truesize([1800 2000]);
%     name = 'Barchart_' + string(f) + '.png';
%     saveas(gcf,name)
% end

figure;
VidObj = VideoWriter('2014.mp4'); %set your file name and video compression
VidObj.FrameRate = 5; %set your frame rate
VidObj.Quality = 100;
open(VidObj);
for f = 1:size(lb, 3)
    I=imagesc(uint8(lb(:,:,f)));
    classNames=[ "BG";"MX";"RH";"IN"];
    ticks = [1.4,2.1,2.9,3.6];
    colorbar('TickLabels',cellstr(classNames),'Ticks',ticks,...
    'TickLength',0, 'TickLabelInterpreter', 'tex')
    map = [0.2 0.4 0.5;1 0 0;0 1 0;0 0 1];
    colormap(map);
    title('2014');
    truesize([3540 3970]);
    frame = getframe(gcf);
    writeVideo(VidObj, frame);
end
close(VidObj);
%%
%test
%clc;
%clear;
%16000:18000,8000:14000
%25400-12000:25400-8000,29700-8000:29700
lb_n = imread('D:\Population_displacement\VHR images\Pleiade_15_16_18\Train_15\lb15gcor10.tif');
lb_n_sub=lb_n(25400-12000:25400-8000,29700-8000:29700);

lb_o = imread('D:\Population_displacement\VHR images\Pleiade_15_16_18\Train_15\lb15g.tif');
lb_o_sub=uint8(lb_o(25400-12000:25400-8000,29700-8000:29700));

im = imread('D:\Population_displacement\VHR images\Pleiade_15_16_18\Pansharpening_nov_15_Clip.tif');
window = im(25400-12000:25400-8000,29700-8000:29700,1:3);
window  = single(window(:,:,1:3));
max_ = max(max(max(window)));
min_ = min(min(min(window)));
window_norm = (window - min_) / (max_ - min_);
window2 = uint16(window_norm * 65536);
hsvImage = rgb2hsv(window2);
% Extract individual color channels.
hChannel = hsvImage(:, :, 1);
sChannel = hsvImage(:, :, 2);
vChannel = hsvImage(:, :, 3);
vChannel2=histeq(vChannel);
% with the old, original h and s channels.
hsvImage2 = cat(3, hChannel, sChannel, vChannel2);
% Convert back to rgb.
window3 = hsv2rgb(hsvImage2);

change=lb_n_sub - lb_o_sub;
change(change~=0)=1;

figure;
subplot(2,4,1:3)
B_o = labeloverlay(window3,lb_o_sub);
imshow(B_o)
title('2015: Before correction')

subplot(2,4,4)
histogram(lb_o_sub)
title('2015: Before correction')

subplot(2,4,5:7)
B_n = labeloverlay(window3,lb_n_sub);
imshow(B_n)
title('2015: After correction')

subplot(2,4,8)
histogram(lb_n_sub)
title('2015: After correction')

figure;
B_c = labeloverlay(window3,change, 'Colormap','autumn');
imshow(B_c)
title('2015: Change')

%%
clc;
%clear;
cor17 = imread('D:\Population_displacement\VHR images\Pleiade_15_16_18\Train_15\lb15gcor.tif');
cor17_=zeros(254, 297, 5);
patchsize=100;
for i=1:patchsize:size(cor17, 1)
   for j=1:patchsize:size(cor17, 2)
       cor17_(((i-1)/patchsize)+1, ... 
           ((j-1)/patchsize)+1) = ...
           mode(mode(cor17(i:i+patchsize-1, j:j+patchsize-1)));
   end
end

% cor17_2 = imread('lb17gc2.tif');
% cor17_2_=zeros(254, 297, 5);
% patchsize=100;
% for i=1:patchsize:size(cor17_2, 1)
%    for j=1:patchsize:size(cor17_2, 2)
%        cor17_2_(((i-1)/patchsize)+1, ... 
%            ((j-1)/patchsize)+1) = ...
%            mode(mode(cor17_2(i:i+patchsize-1, j:j+patchsize-1)));
%    end
% end
% 
% cor17_3 = imread('lb17gc3.tif');
% cor17_3_=zeros(254, 297, 5);
% patchsize=100;
% for i=1:patchsize:size(cor17_3, 1)
%    for j=1:patchsize:size(cor17_3, 2)
%        cor17_3_(((i-1)/patchsize)+1, ... 
%            ((j-1)/patchsize)+1) = ...
%            mode(mode(cor17_3(i:i+patchsize-1, j:j+patchsize-1)));
%    end
% end

lb17 = imread('D:\Population_displacement\VHR images\Pleiade_15_16_18\Train_15\lb15g.tif');
lb17_=zeros(254, 297, 5);
patchsize=100;
for i=1:patchsize:size(lb17, 1)
   for j=1:patchsize:size(lb17, 2)
       lb17_(((i-1)/patchsize)+1, ... 
           ((j-1)/patchsize)+1) = ...
           mode(mode(lb17(i:i+patchsize-1, j:j+patchsize-1)));
   end
end
%%
delta = (lb17_)-(cor17_);
delta(delta~=0)=1;
histogram(delta,'Normalization','probability')

%%
score_15 = imread('label_15.tif');
score_15_by_14 = imread('label_15_by_14.tif');
score_15=reshape(score_15, [25400*29700 4]);
score_15_by_14=reshape(score_15_by_14, [25400*29700 4]);

Rsq1 = 1 - sum((label_15 - label_15_by_14).^2)/sum((label_15 - mean(label_15)).^2)

%% results

class_aug_14 = load('C:\Users\bnikparv\Downloads\models2\patchbased\3class\class_aug_14.mat');
%classPL = load('C:\Users\bnikparv\Downloads\models2\patchbased\3class\class_pl.mat');

cat = categorical({'Background','Mix landuse','High res'});

classSumGE = sum(class_aug_14);
classScaledGE = (class_aug_14 / classSumGE) * 100; 
classSumPL = sum(classPL);
classScaledPL = (classPL / classSumPL) * 100; 

b = [class_aug_14;classPL];
figure;
bb = bar(cat, b)
title('Number of patches per class')
xtips1 = bb(1).XEndPoints;
ytips1 = bb(1).YEndPoints;
labels1 = string(bb(1).YData);
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

xtips2 = bb(2).XEndPoints;
ytips2 = bb(2).YEndPoints;
labels2 = string(bb(2).YData);
text(xtips2,ytips2,labels2,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

legend('2014','2019')

bScaled = [classScaledGE;classScaledPL];
figure;
bar(cat, bScaled)
title('Percentage of patches per class')
legend('2014','2019')

%
cat = categorical({'2014','2019'});

class_aug_14 = histcounts(im_2014)
class_aug_14 = class_aug_14(:, 2:5)/10000;
classSumGE = sum(class_aug_14);
classScaledGE = (class_aug_14 / classSumGE) * 100;

classPL = histcounts(im_2019)
classPL = classPL(:, 2:5)/10000;
classSumPL = sum(classPL);
classScaledPL = (classPL / classSumPL) * 100; 

b = [class_aug_14;classPL];
figure;
bb = bar(cat, b)
title('Number of patches per class')
legend('BG','RH', 'RL', 'RN')

bScaled = [classScaledGE;classScaledPL];
figure;
bb = bar(cat, bScaled)
title('Number of patches per class')
legend('BG','RH', 'RL', 'RN')

xtips1 = bb(1).XEndPoints;
ytips1 = bb(1).YEndPoints;
labels1 = string(bb(1).YData);
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

xtips2 = bb(2).XEndPoints;
ytips2 = bb(2).YEndPoints;
labels2 = string(bb(2).YData);
text(xtips2,ytips2,labels2,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')
%%


