function expName = unet(expName, size, encoderDepth, filters, batchsize, epochs, useDataAugmentation, L2Reg, lr, ...
    gradientclipping, path, splits, folders, savePredictionsFolder, classNames, labelIDs, valPat, isMAT )

% Create a U-Net network
numClasses  = length(classNames);
lgraph = unetLayers(size,numClasses,'EncoderDepth',encoderDepth, 'NumFirstEncoderFilters', filters)

%%     TRAIN       %%
% Load training dataset
if isMAT == true
    imdsTrain = imageDatastore(strcat(path, '/', splits(1), '/', folders(1)), 'FileExtensions','.mat', 'ReadFcn', @loadMAT);
else
    imdsTrain = imageDatastore(strcat(path, '/', splits(1), '/', folders(1)));
end
    pxdsTrain = pixelLabelDatastore(strcat(path, '/', splits(1), '/', folders(2)),classNames,labelIDs);
% Add weights to the classes
    % Get weights
    tbl = countEachLabel(pxdsTrain)

    % UNIFORM PRIOR WEIGHTING
%     prior = 1/numel(classNames);
%     uniformClassWeights = prior./tbl.PixelCount

    % INVERSE FREQUENCY WEIGHTING
%      totalNumberOfPixels = sum(tbl.PixelCount);
%      freq = tbl.PixelCount / totalNumberOfPixels;
%      invFreqClassWeights = 1./freq
     
     % MEDIAN FREQUENCY WEIGHTING
     imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
     medFreqClassWeights = median(imageFreq) ./ imageFreq

% Re-create the layer but with weights added
layer_to_add = [pixelClassificationLayer('Classes',classNames,'ClassWeights',medFreqClassWeights,'Name','Segmentation-Layer')];
% Replace layer
lgraph = replaceLayer(lgraph,'Segmentation-Layer',layer_to_add);
% Display the network.
% analyzeNetwork(lgraph)

% Create a datastore for training the network
if useDataAugmentation == false % NO DATA AUGMENTATION
    ds = pixelLabelImageDatastore(imdsTrain,pxdsTrain);
else % DATA AUGMENTATION
    augmenter = imageDataAugmenter('RandXReflection',true, 'RandYReflection',true)%,'RandRotation',[-10 10], 'RandXTranslation', [-5 5], 'RandYTranslation', [-5 5]);  
    ds = pixelLabelImageDatastore(imdsTrain,pxdsTrain, 'DataAugmentation', augmenter);
end 

% VALIDATION SET
% Load validation dataset.
if isMAT==true
    imdsVal = imageDatastore(strcat(path, '/', splits(3), '/', folders(1)), 'FileExtensions','.mat', 'ReadFcn', @loadMAT);
else
    imdsVal = imageDatastore(strcat(path, '/', splits(3), '/', folders(1)));
end
pxdsVal = pixelLabelDatastore(strcat(path, '/', splits(3), '/', folders(2)),classNames,labelIDs);  
valData = pixelLabelImageDatastore(imdsVal, pxdsVal); 
valFreq = floor(length(ds.Images)/batchsize)

% Set training OPTIONS
options = trainingOptions(...
    'adam', ...    
    ... 'rmsprop' 
    ... 'sgdm','Momentum', 0.9, ...
    'InitialLearnRate',lr, ...
    ... 'LearnRateSchedule','piecewise', ...
    ... 'LearnRateDropFactor', dropfactor, ...
    ... 'LearnRateDropPeriod', 1, ...
    'MaxEpochs',epochs, ...
    'VerboseFrequency',10, ...
    'MiniBatchSize' , batchsize, ...
    ... 'Plots','training-progress', ...
    'L2Regularization',L2Reg, ...,
    'ValidationData',valData, ...,
    'ValidationFrequency', valFreq,...,
    'ValidationPatience', valPat,...,
    ... 'GradientThresholdMethod','l2norm',...
    ... 'GradientThreshold',gradientclipping, ...
    'Shuffle','every-epoch', ...
    'ExecutionEnvironment', 'gpu');

% TRAIN the network
tic;
[net,info] = trainNetwork(ds,lgraph,options)
traintime=toc;

%%         TEST       %%
% Load test dataset.
if isMAT==true
    imdsTest = imageDatastore(strcat(path, '/', splits(2), '/', folders(1)), 'FileExtensions','.mat', 'ReadFcn', @loadMAT);
else
    imdsTest = imageDatastore(strcat(path, '/', splits(2), '/', folders(1)));
end
pxdsTest = pixelLabelDatastore(strcat(path, '/', splits(2), '/', folders(2)),classNames,labelIDs);

%Run the network on the test images. Predicted labels are returned as a pixelLabelDatastore.
tic
pxdsResults = semanticseg(imdsTest,net, 'MiniBatchSize',batchsize,"WriteLocation", savePredictionsFolder);
toc

%Compute Confusion Matrix and Segmentation Metrics(Evaluate the prediction results against the ground truth)
metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest)
metrics.ClassMetrics
metrics.NormalizedConfusionMatrix
metrics.ConfusionMatrix
metrics.DataSetMetrics

% Create dir for the experiment

mkdir(['ExpUNet/',expName]);
% Save matlab snapshot on 'results.mat'
save(['ExpUNet/',expName, '/results'])
% Store metrics in CSV
writetable(metrics.DataSetMetrics,['ExpUNet/',expName,'/dataset.csv'])
writetable(metrics.ClassMetrics, ['ExpUNet/',expName,'/classmetrics.csv'])
writetable(metrics.ConfusionMatrix, ['ExpUNet/',expName,'/confusionmatrix.csv'])
writetable(metrics.NormalizedConfusionMatrix, ['ExpUNet/',expName,'/normconfusionmatrix.csv'])
% Test on 6 images and save a .png
saveTestImages(net, imdsTest, pxdsTest,classNames, ['ExpUNet/',expName,'/ejemplos.png'], labelIDs, isMAT)


% Generate graphics
plotAccuracy(info, options, length(pxdsTrain.Files), 100, ['ExpUNet/',expName]);
plotLoss(info, options, length(pxdsTrain.Files), 20, ['ExpUNet/',expName]);
traintime=datevec(traintime./(60*60*24))
writematrix(traintime, ['ExpUNet/' expName '/tiempo.txt'])

end


function final_matrix = loadMAT(filename)
    load(filename)
end
