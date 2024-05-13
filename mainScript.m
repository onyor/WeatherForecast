% mainScript.m
function mainScript()
    clear;
    clc;

    baseDir = 'C:\Users\onur.yildiz\OneDrive - Logo\Documents\MATLAB\Weather\';
    trainDataPath = fullfile(baseDir, 'dataset');
    testDataPath = fullfile(baseDir, 'testset');

    % imageSizeAlexNet = [227 227 3];
    imageSizeOthers = [224 224 3];

    imds = imageDatastore(trainDataPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames', 'FileExtensions', '.jpg');
    % Veri seti, eğitim ve doğrulama setlerine %70-%30 oranında bölünüyor
    [imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 'randomize');
    % Eğitim ve doğrulama setlerindeki veri sayıları loglanıyor
    trainCounts = countEachLabel(imdsTrain);
    validationCounts = countEachLabel(imdsValidation);
    % Eğitim ve doğrulama veri setlerinin detaylarını gösteren tablo oluşturuluyor
    rowNames = cellstr(trainCounts.Label);
    summaryTable = table(trainCounts.Count, validationCounts.Count, 'RowNames', rowNames, ...
                         'VariableNames', {'TrainingSet', 'ValidationSet'});
    disp('Eğitim ve Doğrulama Veri Seti Detayları:');
    disp(summaryTable);


    % [trainedNetAlexNet, imdsValidationAlexNet] = AlexNet(trainDataPath,imageSizeAlexNet);
    [trainedNetResNet50, imdsValidationResNet50] = ResNet50(trainDataPath,imageSizeOthers);
    % [trainedNetVGG16, imdsValidationVGG16] = VGG16(trainDataPath,imageSizeOthers);
    % [trainedNetGoogleNet, imdsValidationGoogleNet] = GoogleNet(trainDataPath,imageSizeOthers);

    % showImageProcessing(trainDataPath);
   
    % validationAccuracyAlexNet = performValidation(trainedNetAlexNet, imdsValidationAlexNet,imageSizeAlexNet);
    % fprintf('AlexNet: Doğrulama Seti Sınıflandırma Başarı Oranı: %.2f%%\n', validationAccuracyAlexNet);

    validationAccuracyResNet50 = performValidation(trainedNetResNet50, imdsValidationResNet50, imageSizeOthers);
    fprintf('ResNet50: Doğrulama Seti Sınıflandırma Başarı Oranı: %.2f%%\n', validationAccuracyResNet50);   

    % validationAccuracyVGG16 = performValidation(trainedNetVGG16, imdsValidationVGG16, imageSizeOthers);
    % fprintf('VGG16: Doğrulama Seti Sınıflandırma Başarı Oranı: %.2f%%\n', validationAccuracyVGG16);
    % 
    % validationAccuracyGoogleNet = performValidation(trainedNetGoogleNet, imdsValidationGoogleNet, imageSizeOthers);
    % fprintf('GoogleNet: Doğrulama Seti Sınıflandırma Başarı Oranı: %.2f%%\n', validationAccuracyGoogleNet);


    % Doğrulama sonuçlarını görselleştirme
    showValidationResults(trainDataPath, imageSizeOthers, trainedNetResNet50);


    % testAccuracyAlexNet = performTest(trainedNetAlexNet, testDataPath, imageSizeAlexNet);
    % fprintf('AlexNet: Test Seti Sınıflandırma Başarı Oranı: %.2f%%\n', testAccuracyAlexNet);
  
    % testAccuracyResNet50 = performTest(trainedNetResNet50, testDataPath, imageSizeOthers);
    % fprintf('ResNet50: Test Seti Sınıflandırma Başarı Oranı: %.2f%%\n', testAccuracyResNet50);

    % testAccuracyVGG16 = performTest(trainedNetVGG16, testDataPath, imageSizeOthers);
    % fprintf('VGG16: Test Seti Sınıflandırma Başarı Oranı: %.2f%%\n', testAccuracyVGG16);
    % 
    % testAccuracyGoogleNet = performTest(trainedNetGoogleNet, testDataPath, imageSizeOthers);
    % fprintf('GoogleNet: Test Seti Sınıflandırma Başarı Oranı: %.2f%%\n', testAccuracyGoogleNet);
end





