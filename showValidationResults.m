function showSelectedPredictions(dataFolderPath, imageSize, trainedNet)
    % Veri seti yolundan doğrulama setini yükleyin
    imds = imageDatastore(dataFolderPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames', 'FileExtensions', '.jpg');
    [~, imdsValidation] = splitEachLabel(imds, 0.7, 'randomize');  % %70 eğitim, %30 doğrulama

    % Doğrulama setini augmentedImageDatastore ile hazırlayın
    augimdsValidation = augmentedImageDatastore(imageSize, imdsValidation, 'ColorPreprocessing', 'gray2rgb');

    % Tahminleri gerçekleştirin ve gerçek etiketlerle karşılaştırın
    [predictions, scores] = classify(trainedNet, augimdsValidation);
    actualLabels = imdsValidation.Labels;

    % Tahmin sonuçlarını tablo olarak oluşturun
    resultsTable = table(imdsValidation.Files, actualLabels, predictions, 'VariableNames', {'File', 'TrueLabel', 'PredictedLabel'});

    % Yanlış tahmin edilenleri bulun ve rastgele 8 tanesini seçin
    incorrectPredictions = resultsTable(resultsTable.TrueLabel ~= resultsTable.PredictedLabel, :);
    numIncorrectToShow = min(8, size(incorrectPredictions, 1));
    incorrectIndices = randperm(size(incorrectPredictions, 1), numIncorrectToShow);

    % Doğru tahmin edilenleri bulun ve rastgele 8 tanesini seçin
    correctPredictions = resultsTable(resultsTable.TrueLabel == resultsTable.PredictedLabel, :);
    numCorrectToShow = min(8, size(correctPredictions, 1));
    correctIndices = randperm(size(correctPredictions, 1), numCorrectToShow);

    % Yanlış tahmin edilen resimleri göster
    figure('Name', 'Yanlış Tahminler');
    for i = 1:numIncorrectToShow
        subplot(ceil(numIncorrectToShow/4), 4, i);
        img = imread(incorrectPredictions.File{incorrectIndices(i)});
        imshow(img);
        title(['Tahmin: ' char(incorrectPredictions.PredictedLabel(incorrectIndices(i))) ...
               ', Doğrusu: ' char(incorrectPredictions.TrueLabel(incorrectIndices(i)))]);
    end

    % Doğru tahmin edilen resimleri göster
    figure('Name', 'Doğru Tahminler');
    for i = 1:numCorrectToShow
        subplot(ceil(numCorrectToShow/4), 4, i);
        img = imread(correctPredictions.File{correctIndices(i)});
        imshow(img);
        title(['Tahmin: ' char(correctPredictions.PredictedLabel(correctIndices(i))) ...
               ', Doğrusu: ' char(correctPredictions.TrueLabel(correctIndices(i)))]);
    end
end
