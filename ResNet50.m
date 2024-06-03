function [net, imdsValidation] = ResNet50(dataFolderPath, imageSize)
    % Veri yolu tanımlanıyor ve image datastore oluşturuluyor
    imds = imageDatastore(dataFolderPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames', 'FileExtensions', '.jpg');

    % Veri seti, eğitim ve doğrulama setlerine %70-%30 oranında bölünüyor
    [imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 'randomize');

    % Eğitim ve doğrulama setlerindeki veri sayıları loglanıyor
    trainCounts = countEachLabel(imdsTrain);
    validationCounts = countEachLabel(imdsValidation);

    % Bu adımda, görüntüleri artırmak için imageDataAugmenter nesnesi oluşturuluyor.
    % Görüntülerin rastgele döndürülmesi, kaydırılması, ölçeklendirilmesi ve yansıtılması gibi çeşitli veri artırma işlemleri tanımlanıyor.
    imageAugmenter = imageDataAugmenter( ...
        'RandRotation', [-30, 30], ... % Görüntülerin rastgele -30 ile 30 derece arasında döndürülmesi
        'RandXTranslation', [-10 10], ... % Görüntülerin rastgele yatayda -10 ile 10 piksel arasında kaydırılması
        'RandYTranslation', [-10 10], ... % Görüntülerin rastgele dikeyde -10 ile 10 piksel arasında kaydırılması
        'RandXScale', [0.75 1.25], ... % Görüntülerin yatayda rastgele %75 ile %125 arasında ölçeklendirilmesi
        'RandYScale', [0.75 1.25], ... % Görüntülerin dikeyde rastgele %75 ile %125 arasında ölçeklendirilmesi
        'RandXReflection', true, ... % Görüntülerin yatayda yansıtılması
        'RandYReflection', true); % Görüntülerin dikeyde yansıtılması
    
    % Eğitim veri kümesi için augmented image datastore oluşturuluyor
    augimdsTrain = augmentedImageDatastore(imageSize, imdsTrain, ...
        'DataAugmentation', imageAugmenter, ... % Veri artırma işlemleri uygulanıyor
        'ColorPreprocessing', 'gray2rgb'); % Gri tonlamalı görüntüler RGB'ye dönüştürülüyor
    
    % Doğrulama veri kümesi için augmented image datastore oluşturuluyor
    augimdsValidation = augmentedImageDatastore(imageSize, imdsValidation, ...
        'ColorPreprocessing', 'gray2rgb'); % Gri tonlamalı görüntüler RGB'ye dönüştürülüyor
    
    % ResNet-50 modelini yükleyin ve son katmanlarını özelleştirin
    net = resnet50;
    
    % Modelin katman grafiğini alın
    lgraph = layerGraph(net);
    numClasses = numel(categories(imdsTrain.Labels));
    
    % Yeni katmanları tanımlayın
    newLayers = [
        fullyConnectedLayer(numClasses, 'Name', 'new_fc', ... % Yeni tam bağlantılı katman, sınıf sayısına göre ayarlanır
        'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10), ...
        softmaxLayer('Name', 'new_softmax'), ... % Softmax katmanı
        classificationLayer('Name', 'new_classoutput')]; % Sınıflandırma katmanı
    
    % Yeni katmanlarla eski katmanları değiştirin
    lgraph = replaceLayer(lgraph, 'fc1000', newLayers(1)); % 'fc1000' katmanını yeni tam bağlantılı katmanla değiştirin
    lgraph = replaceLayer(lgraph, 'fc1000_softmax', newLayers(2)); % 'fc1000_softmax' katmanını yeni softmax katmanıyla değiştirin
    lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000', newLayers(3)); % 'ClassificationLayer_fc1000' katmanını yeni sınıflandırma katmanıyla değiştirin
    
    % Eğitim seçeneklerini belirleyin
    options = trainingOptions('adam', ... % Adam optimizasyon algoritması kullanılıyor
        'ExecutionEnvironment', 'cpu', ... % Eğitim CPU üzerinde gerçekleştirilecek
        'InitialLearnRate', 0.0001, ... % Başlangıç öğrenme hızı
        'MaxEpochs', 10, ... % Maksimum epoch sayısı
        'MiniBatchSize', 32, ... % Mini batch boyutu
        'Shuffle', 'every-epoch', ... % Veriler her epochta karıştırılacak
        'ValidationData', augimdsValidation, ... % Doğrulama verileri belirtiliyor
        'Plots', 'training-progress', ... % Eğitim ilerleme grafikleri gösterilecek
        'Verbose', true); % Eğitim sırasında detaylı bilgi verilecek



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
