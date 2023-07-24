%%
clc;
clear;
modelfile = 'G:/backupC27152020/C/models2/matlab/net_12_8_20_2.onnx';
classes = ["BG" "MX" "RH" "RN"];
net = importONNXNetwork(modelfile,'OutputLayerType','classification','Classes',classes)

%%
img_ = imread('G:/backupC27152020/D/Population_displacement/VHRimages/finalProcessedImages/vhraug18matched.tif');
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
       
       %max_ = max(max(max(window_rs)));
       %min_ = min(min(min(window_rs)));
       max_ = 6926;
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

%imshow(label_(:, :, 1))
imwrite(uint8(label_(:,:,1)), 'G:/backupC27152020/D/Population_displacement/VHRimages/finalProcessedImages/vhr_Xception_18.tif');

