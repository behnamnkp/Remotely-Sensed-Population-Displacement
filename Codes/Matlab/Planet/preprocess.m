clc;
clear;
%%
listing = dir(...
    'D:\Population_displacement\VHR images\Pleiade_15_16_18\Train_18_aug\RH')

for i=3:size(listing, 1)
    baseFileName = listing(i).name
    if contains(baseFileName, '.xml') || contains(baseFileName, '.ovr') || contains(baseFileName, 'tfw') 
       delete(baseFileName); 
    end
end
%%
clc
clear
listing = dir(...
    'C:\Users\bnikparv\Downloads\VHRClassified\VHRtrainingsamples\allbck\RN')

for i=3:size(listing, 1)
    baseFileName = listing(i).name
    if contains(baseFileName, '.xml') || contains(baseFileName, '.ovr') || contains(baseFileName, '.tfw') || contains(baseFileName, '.aux') 
    else
        img = imread(listing(i).name);
        img  = single(img(:,:,1:3));
        
        if img(end,:,1)==-9999
            img(end,:,:)=[];
        end
        if img(1,:,1)==-9999
            img(1,:,:)=[]; 
        end
        if img(:,1,1)==-9999
            img(:,1,:)=[];
        end
        if img(:,end,1)==-9999
            img(:,end,:)=[];  
        end
        
        if img(end,:,2)==-9999
            img(end,:,:)=[];
        end
        if img(1,:,2)==-9999
            img(1,:,:)=[]; 
        end
        if img(:,1,2)==-9999
            img(:,1,:)=[];
        end
        if img(:,end,2)==-9999
            img(:,end,:)=[];  
        end
        
        if img(end,:,3)==-9999
            img(end,:,:)=[];
        end
        if img(1,:,3)==-9999
            img(1,:,:)=[]; 
        end
        if img(:,1,3)==-9999
            img(:,1,:)=[];
        end
        if img(:,end,3)==-9999
            img(:,end,:)=[];  
        end

        a_rs = imresize(img, [299 299]);
        max_ = max(max(max(img)));
        min_ = min(min(min(img)));

        %max_ = 6926;
        %min_ = 0;

        a_norm = (a_rs - min_) / (max_ - min_);
        a_norm2 = uint16(a_norm * 65536);
        

        %postFileName = replace(baseFileName,'.TIF','.png')
        imwrite(a_norm2, baseFileName);
        i
    end
end
%% all zero check
clc
clear
listing = dir(...
    'C:\Users\bnikparv\Downloads\VHRClassified\VHRtrainingsamples\allbck\MX')

l = 0
for i=3:size(listing, 1)
    baseFileName = listing(i).name;
    if contains(baseFileName, '.xml') || contains(baseFileName, '.ovr') || contains(baseFileName, '.tfw') || contains(baseFileName, '.aux') 
    else
        img = imread(listing(i).name);
        if all(img(:)==0)
            listing(i).name
            delete(baseFileName)
            l = l + 1
        end
        
        i;
    end
end


%% min max
clc
clear
listing = dir(...
    'C:\Users\bnikparv\Downloads\VHRClassified\VHRtrainingsamples\all\BG')

max_all = 0;
min_all = 0;
for i=3:size(listing, 1)
    baseFileName = listing(i).name;
    if contains(baseFileName, '.xml') || contains(baseFileName, '.ovr') || contains(baseFileName, '.tfw') || contains(baseFileName, '.aux') 
    else
        img = imread(listing(i).name);
        img  = img(:,:,1:3);

        if img(end,:,1)==-9999
            img(end,:,:)=[];
        end
        if img(1,:,1)==-9999
            img(1,:,:)=[]; 
        end
        if img(:,1,1)==-9999
            img(:,1,:)=[];
        end
        if img(:,end,1)==-9999
            img(:,end,:)=[];  
        end
        
        if img(end,:,2)==-9999
            img(end,:,:)=[];
        end
        if img(1,:,2)==-9999
            img(1,:,:)=[]; 
        end
        if img(:,1,2)==-9999
            img(:,1,:)=[];
        end
        if img(:,end,2)==-9999
            img(:,end,:)=[];  
        end
        
        if img(end,:,3)==-9999
            img(end,:,:)=[];
        end
        if img(1,:,3)==-9999
            img(1,:,:)=[]; 
        end
        if img(:,1,3)==-9999
            img(:,1,:)=[];
        end
        if img(:,end,3)==-9999
            img(:,end,:)=[];  
        end
        
        max_ = max(max(max(img)));
        min_ = min(min(min(img)));
        
        if min_ == -9999
            delete(baseFileName)
        end

    end
    if max_all < max_
        max_all = max_
    end
    if min_all > min_ & min_ ~= -9999
        min_all = min_
    end
end


%%
%rename
listing = dir(...
    'G:\backupC27152020\Population_Displacement_Final\Resources\VHR\training_patches_png\BG')

for i=3:size(listing, 1)
    baseFileName = listing(i).name;
    img = imread(listing(i).name);
    postFileName = replace(baseFileName,'.TIF','.jpg')
    imwrite(img, postFileName);
    delete(baseFileName)
    i
end

%%
listing = dir(...
    'C:\Users\bnikparv\Downloads\VHRClassified\VHRtrainingsamples\train16')

for i=3:size(listing, 1)
    if contains(baseFileName, '.xml') || contains(baseFileName, '.ovr') || contains(baseFileName, 'tfw')
    else
        baseFileName = listing(i).name;
        img = imread(listing(i).name);
        img  = img(:,:,1:3);
        imwrite(uint16(img), baseFileName);
        i
    end
end

%%

listing = dir(...
    'D:\Population_displacement\population_data\LandScanData\study_area_clp')

for i=3:size(listing, 1)
    if contains(baseFileName, '.xml') || contains(baseFileName, '.ovr') || contains(baseFileName, 'tfw') || 
        contains(baseFileName, 'clip') || contains(baseFileName, 'nrm') || contains(baseFileName, 'rsp')
    else
        baseFileName = listing(i).name;
        img = imread(listing(i).name);
        if img(:,:)==-9999
            img(:,:)=0;
        end
        img  = reshape(size(img, 1)*size(img, 2), 1)
        sum()
        imwrite(uint16(img), baseFileName);
        i
    end
end
