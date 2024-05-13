function [net, imdsValidation] = GoogleNet(dataFolderPath, imageSize)
    imds = imageDatastore(dataFolderPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames', 'FileExtensions', '.jpg');
    [imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 'randomize');
    
    imageAugmenter = imageDataAugmenter( ...
        'RandRotation', [-30, 30], ...
        'RandXTranslation', [-10 10], ...
        'RandYTranslation', [-10 10], ...
        'RandXScale', [0.75 1.25], ...
        'RandYScale', [0.75 1.25], ...
        'RandXReflection', true, ...
        'RandYReflection', true);
    augimdsTrain = augmentedImageDatastore(imageSize, imdsTrain, 'DataAugmentation', imageAugmenter, 'ColorPreprocessing', 'gray2rgb');
    augimdsValidation = augmentedImageDatastore(imageSize, imdsValidation, 'ColorPreprocessing', 'gray2rgb');

    net = googlenet;  % Doğru modeli yükleyin
    lgraph = layerGraph(net);
    numClasses = numel(categories(imdsTrain.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses, 'Name', 'new_fc', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10),
        softmaxLayer('Name', 'new_softmax'),
        classificationLayer('Name', 'new_classoutput')];
    lgraph = replaceLayer(lgraph, 'loss3-classifier', newLayers(1)); % GoogleNet'e özgü son sınıflandırma katmanı
    lgraph = replaceLayer(lgraph, 'prob', newLayers(2));
    lgraph = replaceLayer(lgraph, 'output', newLayers(3));

    options = trainingOptions('adam', ...
        'ExecutionEnvironment', 'cpu', ...
        'InitialLearnRate', 0.0001, ...
        'MaxEpochs', 10, ...
        'MiniBatchSize', 32, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', augimdsValidation, ...
        'Plots', 'training-progress', ...
        'Verbose', true);

    [net, info] = trainNetwork(augimdsTrain, lgraph, options);
end
