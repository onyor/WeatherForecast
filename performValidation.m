% performValidation.m
function validationAccuracy = performValidation(net, imdsValidation, imageSize)
    augimdsValidation = augmentedImageDatastore(imageSize, imdsValidation, 'ColorPreprocessing', 'gray2rgb');
    [validationYPred, ~] = testModel(net, augimdsValidation);
    validationAccuracy = sum(validationYPred == imdsValidation.Labels) / numel(imdsValidation.Labels) * 100;
end