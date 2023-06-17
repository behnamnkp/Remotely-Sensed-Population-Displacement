%% Train the Xception network
clc;
clear;
cd 'G:\backupC27152020\Population_Displacement_Final\'
%% Read train data
%gpuDevice
% path to the root file of training data
digitDatasetPath = fullfile(...
    'G:\backupC27152020\Population_Displacement_Final\Resources\VHR\training_patches\');

imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

rng(1234);
train_data = imds;
[train_data,imdsValidation] = splitEachLabel(train_data,0.7);
[imdsTrain,imdsVal] = splitEachLabel(train_data,0.8);

%addpath('/users/bnikparv/Documents/MATLAB/Examples/R2019b/nnet/TransferLearningUsingGoogLeNetExample')
 
%% Count of datasets
labelCountTrain = countEachLabel(imdsTrain)
labelCountVal = countEachLabel(imdsVal)
labelCountTvalidation = countEachLabel(imdsValidation)

%% Read image patches
img = readimage(imdsTrain,1);
size(img)

net = xception;

inputSize = [299 299 3];
%imdsTrain.ReadFcn = @(loc)imresize(imread(loc),inputSize);
%imdsVal.ReadFcn = @(loc)imresize(imread(loc),inputSize);
%imdsValidation.ReadFcn = @(loc)imresize(imread(loc),inputSize);

numClasses = numel(categories(imdsTrain.Labels))

%% Adjust the Xception network for four class classification
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
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

% Transfer learning
layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:34) = freezeWeights(layers(1:34));
lgraph = createLgraphUsingConnections(layers,connections);

%load('/users/bnikparv/matlab/all/net_checkpoint__10075__2020_10_31__11_28_11.mat')

%% Set the options for the network
%     'ExecutionEnvironment','multi-gpu',...
options = trainingOptions('adam', ...
    'MiniBatchSize',16, ...
    'MaxEpochs',3, ...
    'InitialLearnRate',1e-3, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsVal , ...
    'ValidationFrequency',3, ...
    'Verbose',true, ...
    'CheckpointPath','G:\backupC27152020\Population_Displacement_Final\Code\Matlab\checkpoints\', ...
    'Plots','training-progress')

%%
%tic
netTransfer= trainNetwork(imdsTrain,lgraph,options)
%netTransfer= trainNetwork(imdsTrain,layerGraph(net),options)
%toc

filename = 'G:\backupC27152020\Population_Displacement_Final\Code\Matlab\Xception\net_03262021.onnx';
exportONNXNetwork(netTransfer,filename)
save(['net_03262021.mat'],'netTransfer','options');

%modelfile = '/users/bnikparv/matlab/planetImage/train15/plane15xXception.onnx';
%classes = ["BG","MX","RH", "RN"];
%net = importONNXNetwork(modelfile,'OutputLayerType','classification','Classes',classes)

%% Classify the validation data
YPred = classify(netTransfer,imdsValidation);
YValidation = imdsValidation.Labels;

figure('Units','normalized','Position',[0.2 0.2 0.4 0.4]);
cm = confusionchart(YValidation,YPred);
cm.NormalizedValues
cm.Title = 'Confusion Matrix for Validation Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

accuracy = sum(YPred == YValidation)/numel(YValidation)
save('/Code/Matlab/Xception/variables_date.mat','cm','YPred', 'YValidation', 'imdsTrain', 'imdsValidation','imdsVal','-v7.3');

%% Classify the entire image
clc;
clear;
addpath('./Resources/VHR/Network')
modelfile = 'net_10_8_20.onnx';
classes = ["BG" "MX" "RH" "RN"];
net = importONNXNetwork(modelfile,'OutputLayerType','classification','Classes',classes)

% Read image
addpath('./Resources/VHR/images')
img_ = imread('vhrjun17matched.tif');
img_ = img_(:,:,1:3);

size(img_)
windowSize = 100

nrow=25201
ncol=29501

label_ = zeros([nrow ncol 4]);

bgCount = 0;
rhCount = 0;
mxCount = 0;
rnCount = 0;

for i=1:windowSize:nrow
   for j=1:windowSize:ncol 
       
       window = img_(i:i+windowSize-1, j:j+windowSize-1, :);
       window  = single(window(:,:,1:3));
       window_rs = imresize(window, [299 299]);
       
%        max_ = max(max(max(window_rs)));
%        min_ = min(min(min(window_rs)));
       max_ = 4510;
       min_ = 0;

       window_norm = (window_rs - min_) / (max_ - min_);
       %max_n = max(max(max(window_norm)));
       %min_n = min(min(min(window_norm)));

       window2 = uint16(window_norm * 65536);
       
       [label,scores] = classify(net,window2);
       
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
       elseif label == 'RH'
           label_(i:i+windowSize-1, j:j+windowSize-1, 1) = 3;
           rhCount = rhCount + 1;
       else
           label_(i:i+windowSize-1, j:j+windowSize-1, 1) = 4;
           rhCount = rhCount + 1;
       end
       
       label_(i:i+windowSize-1, j:j+windowSize-1, 2) = scoresTop(1);
       
       label;
   end
   i
end

imwrite(uint8(label_(:,:,1)), '/Data/VHR/images/label.tif');

%% georeference labels
img = imread('vhr_Xception_18.tif');
a = uint8(img);
% open a file for writing
fid = fopen('label2018.txt', 'wt');
% print a title, followed by a blank line
% 
fprintf(fid, 'NCOLS 29600\nNROWS 25300\nXLLCORNER 325355.593829\nYLLCORNER 4018639.81494\n');
fprintf(fid, 'CELLSIZE 0.5\nNODATA_VALUE 0\n');
% fprintf(fid, 'NCOLS 29630\nNROWS 25353\nXLLCORNER 325355.95\nYLLCORNER 4031346.05\n');
% fprintf(fid, 'CELLSIZE 0.499991422727483\nNODATA_VALUE 0\n');
for ii = 1:size(a,1)
    fprintf(fid,'%g ',a(ii,:));
    fprintf(fid,'\n');
end
fclose(fid);

%% Create tif in GIS
%Ascii to raster

%% Extract features
% 9-10-11-23
layer = 26;
name = net.Layers(layer).Name

channels = 1:256;
% net.Layers(end).Classes(channels)
I = deepDreamImage(net,name,channels, 'Verbose',false, ...
    'NumIterations',100, 'PyramidLevels',2)
figure
I = imtile(I,'ThumbnailSize',[150 150]);
imshow(I)
title(['Layer ',name,' Features'],'Interpreter','none')


