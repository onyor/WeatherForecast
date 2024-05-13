function [YPred, scores] = testModel(net, input)
    if ~isa(input, 'augmentedImageDatastore')
        imageSize = [224 224 3];
        imds = imageDatastore(input, 'IncludeSubfolders', true, 'FileExtensions', '.jpg');
        augimds = augmentedImageDatastore(imageSize, imds, 'ColorPreprocessing', 'gray2rgb');

        % Loglama: Test veri setinin detayları
        testCounts = countEachLabel(imds);
        disp('Test veri seti detayları:');
        disp(testCounts);
    else
        augimds = input;  % Bu durumda, augimds'in altında yatan imds'e erişemeyiz, bu nedenle loglama yapılmayacak.
    end
    
    [YPred, scores] = classify(net, augimds);
end
