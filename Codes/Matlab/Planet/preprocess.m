%% Create test and training data sets
clc;
clear;

%% High residential buildings
rh_folder = 'D:\Population_displacement\VHR images\Pleiade_15_16_18\Train_18_aug\RH';
rh_listing = dir(rh_folder);

for i = 3:size(rh_listing, 1)
    baseFileName = rh_listing(i).name;
    if contains(baseFileName, '.xml') || contains(baseFileName, '.ovr') || contains(baseFileName, 'tfw')
        delete(fullfile(rh_folder, baseFileName));
    end
end

%% Non-residential buildings
rn_folder = 'C:\Users\bnikparv\Downloads\VHRClassified\VHRtrainingsamples\allbck\RN';
rn_listing = dir(rn_folder);

for i = 3:size(rn_listing, 1)
    baseFileName = rn_listing(i).name;
    if contains(baseFileName, '.xml') || contains(baseFileName, '.ovr') || contains(baseFileName, '.tfw') || contains(baseFileName, '.aux')
        continue;
    else
        img = imread(fullfile(rn_folder, baseFileName));
        img  = single(img(:,:,1:3));

        % Remove pixels with value -9999
        img(img == -9999) = NaN;
        img = fillmissing(img, 'linear');

        % Normalize the image to the range [0, 1]
        img = (img - min(img(:))) / (max(img(:)) - min(img(:)));

        % Resize the image to [299, 299]
        a_rs = imresize(img, [299, 299]);

        % Convert to uint16 for saving
        a_norm2 = uint16(a_rs * 65536);

        % Save the normalized image
        imwrite(a_norm2, fullfile(rn_folder, baseFileName));
    end
end

%% Mixed residential buildings
mx_folder = 'C:\Users\bnikparv\Downloads\VHRClassified\VHRtrainingsamples\allbck\MX';
mx_listing = dir(mx_folder);

for i = 3:size(mx_listing, 1)
    baseFileName = mx_listing(i).name;
    if contains(baseFileName, '.xml') || contains(baseFileName, '.ovr') || contains(baseFileName, '.tfw') || contains(baseFileName, '.aux')
        continue;
    else
        img = imread(fullfile(mx_folder, baseFileName));
        if all(img(:) == 0)
            listing(i).name
            delete(fullfile(mx_folder, baseFileName));
        end
    end
end

%% Background class
bg_folder = 'C:\Users\bnikparv\Downloads\VHRClassified\VHRtrainingsamples\all\BG';
bg_listing = dir(bg_folder);

max_all = 0;
min_all = 0;
for i = 3:size(bg_listing, 1)
    baseFileName = bg_listing(i).name;
    if contains(baseFileName, '.xml') || contains(baseFileName, '.ovr') || contains(baseFileName, '.tfw') || contains(baseFileName, '.aux')
        continue;
    else
        img = imread(fullfile(bg_folder, baseFileName));
        img  = img(:,:,1:3);

        % Remove pixels with value -9999
        img(img == -9999) = NaN;
        img = fillmissing(img, 'linear');

        max_ = max(img(:));
        min_ = min(img(:));

        if min_ == -9999
            delete(fullfile(bg_folder, baseFileName))
        end

    end
    if max_all < max_
        max_all = max_;
    end
    if min_all > min_ && min_ ~= -9999
        min_all = min_;
    end
end

% Additional code for renaming files...

%% Code for other sections...

% Save the results for BG folder
bg_folder = 'G:\backupC27152020\Population_Displacement_Final\Resources\VHR\training_patches_png\BG';
bg_listing = dir(bg_folder);

for i = 3:size(bg_listing, 1)
    baseFileName = bg_listing(i).name;
    img = imread(fullfile(bg_folder, baseFileName));

    % Replace '.TIF' with '.jpg' in the filename
    postFileName = replace(baseFileName, '.TIF', '.jpg');

    % Save the image in JPG format
    imwrite(img, fullfile(bg_folder, postFileName));

    % Delete the original TIF file
    delete(fullfile(bg_folder, baseFileName));

    % Display the iteration number (optional)
    disp(['Processed image ', num22str(i-2), ' out of ', num2str(size(bg_listing, 1)-2)]);
end

train_folder = 'C:\Users\bnikparv\Downloads\VHRClassified\VHRtrainingsamples\train16';
train_listing = dir(train_folder);

for i = 3:size(train_listing, 1)
    baseFileName = train_listing(i).name;

    % Check if the file has extensions that should be skipped
    if contains(baseFileName, '.xml') || contains(baseFileName, '.ovr') || contains(baseFileName, 'tfw')
        continue;
    else
        img = imread(fullfile(train_folder, baseFileName));
        img = img(:,:,1:3);

        % Save the image as uint16 format
        imwrite(uint16(img), fullfile(train_folder, baseFileName));

        % Display the iteration number (optional)
        disp(['Processed image ', num2str(i-2), ' out of ', num2str(size(train_listing, 1)-2)]);
    end
end

study_folder = 'D:\Population_displacement\population_data\LandScanData\study_area_clp';
study_listing = dir(study_folder);

for i = 3:size(study_listing, 1)
    baseFileName = study_listing(i).name;

    % Check if the file has extensions that should be skipped
    if contains(baseFileName, '.xml') || contains(baseFileName, '.ovr') || contains(baseFileName, 'tfw') || ...
        contains(baseFileName, 'clip') || contains(baseFileName, 'nrm') || contains(baseFileName, 'rsp')
        continue;
    else
        img = imread(fullfile(study_folder, baseFileName));

        % Replace -9999 with 0 in the image
        img(img == -9999) = 0;

        % Reshape the image
        img = reshape(img, [size(img, 1) * size(img, 2), 1]);

        % Calculate the sum (optional, not used further)
        sum_img = sum(img);

        % Save the image as uint16 format
        imwrite(uint16(img), fullfile(study_folder, baseFileName));

        % Display the iteration number (optional)
        disp(['Processed image ', num2str(i-2), ' out of ', num2str(size(study_listing, 1)-2)]);
    end
end

