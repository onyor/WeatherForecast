% performTest.m
function testAccuracy = performTest(net, testDataPath, imageSize)
    imdsTest = imageDatastore(testDataPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    augimdsTest = augmentedImageDatastore(imageSize, imdsTest, 'ColorPreprocessing', 'gray2rgb');
    [testYPred, ~] = testModel(net, augimdsTest);
    testAccuracy = sum(testYPred == imdsTest.Labels) / numel(imdsTest.Labels) * 100;
end
