function [net, imdsValidation] = ResNet50(dataFolderPath, imageSize)
    % Veri yolu tanımlanıyor ve image datastore oluşturuluyor
    imds = imageDatastore(dataFolderPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames', 'FileExtensions', '.jpg');

    % Veri seti, eğitim ve doğrulama setlerine %70-%30 oranında bölünüyor
    [imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 'randomize');

    % Eğitim ve doğrulama setlerindeki veri sayıları loglanıyor
    trainCounts = countEachLabel(imdsTrain);
    validationCounts = countEachLabel(imdsValidation);

    % Görüntü boyutları tanımlanıyor ve augmented image datastore oluşturuluyor
    imageAugmenter = imageDataAugmenter( ...
        'RandRotation', [-30, 30], ...
        'RandXTranslation', [-10 10], ...
        'RandYTranslation', [-10 10], ...
        'RandXScale', [0.75 1.25], ...
        'RandYScale', [0.75 1.25], ...
        'RandXReflection', true, ...
        'RandYReflection', true);
    augimdsTrain = augmentedImageDatastore(imageSize, imdsTrain, ...
        'DataAugmentation', imageAugmenter, 'ColorPreprocessing', 'gray2rgb');
    augimdsValidation = augmentedImageDatastore(imageSize, imdsValidation, 'ColorPreprocessing', 'gray2rgb');

    % ResNet-50 modelini yükleyin ve son katmanlarını özelleştirin
    net = resnet50;
    lgraph = layerGraph(net);
    numClasses = numel(categories(imdsTrain.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses, 'Name', 'new_fc', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10),
        softmaxLayer('Name', 'new_softmax'),
        classificationLayer('Name', 'new_classoutput')];
    lgraph = replaceLayer(lgraph, 'fc1000', newLayers(1));
    lgraph = replaceLayer(lgraph, 'fc1000_softmax', newLayers(2));
    lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000', newLayers(3));

    options = trainingOptions('adam', ...
        'ExecutionEnvironment', 'cpu', ... 
        'InitialLearnRate', 0.0001, ...
        'MaxEpochs', 10, ...
        'MiniBatchSize', 32, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', augimdsValidation, ...
        'Plots', 'training-progress', ...
        'Verbose', true);

    % Sinir ağı eğitiliyor
    [net, info] = trainNetwork(augimdsTrain, lgraph, options);

    % Modelin doğrulama setinde sınıflandırma yapması
    augimdsValidation = augmentedImageDatastore(imageSize, imdsValidation, 'ColorPreprocessing', 'gray2rgb');
    [predictedLabels, ~] = classify(net, augimdsValidation);
    trueLabels = imdsValidation.Labels;
    
    % Karmaşıklık matrisini oluşturma ve gösterme
    figure;
    cm = confusionchart(trueLabels, predictedLabels);
    cm.Title = 'Confusion Matrix for ResNet50';
    cm.ColumnSummary = 'column-normalized';
    cm.RowSummary = 'row-normalized';
end
