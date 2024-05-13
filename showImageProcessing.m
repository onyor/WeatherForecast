function showDetailedImageProcessing(dataFolderPath)
    % Veri yolu tanımlanıyor ve image datastore oluşturuluyor
    imds = imageDatastore(dataFolderPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames', 'FileExtensions', '.jpg');

    % Eğitim ve doğrulama setlerine ayrılıyor
    [imdsTrain, ~] = splitEachLabel(imds, 0.7, 'randomize');

    % Rastgele bir resim seçiliyor
    randomSample = imdsTrain.readimage(randi(numel(imdsTrain.Files)));

    % Original image
    figure;
    imshow(randomSample);
    title('Original Image');

    % Resizing the image
    resizedImage = imresize(randomSample, [224 224]);
    figure;
    imshow(resizedImage);
    title('Resized Image');

    % Converting to grayscale then to RGB
    grayImage = rgb2gray(resizedImage);
    rgbImage = cat(3, grayImage, grayImage, grayImage);
    figure;
    imshow(rgbImage);
    title('Grayscale converted to RGB');

    % Applying data augmentation
    imageAugmenter = imageDataAugmenter(...
        'RandRotation', [-20, 20], ...
        'RandXTranslation', [-5 5], ...
        'RandYTranslation', [-5 5], ...
        'RandXScale', [0.8 1.2], ...
        'RandYScale', [0.8 1.2], ...
        'RandXReflection', true, ...
        'RandYReflection', true);
    augimds = augmentedImageDatastore([224 224 3], rgbImage, 'DataAugmentation', imageAugmenter);

    % Displaying augmented image
    augmentedImages = augimds.read();  % Read augmented images
    
    % Check if the result is a table and extract the image correctly
    if istable(augmentedImages)
        if iscell(augmentedImages.input{1})
            augmentedImage = augmentedImages.input{1}{1};
        else
            augmentedImage = augmentedImages.input{1};
        end
    elseif iscell(augmentedImages)
        augmentedImage = augmentedImages{1}{1};
    else
        augmentedImage = augmentedImages;
    end
    
    % Display the image
    figure;
    imshow(augmentedImage);
    title('Augmented Image');
end
