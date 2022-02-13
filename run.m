%%
clear
close all

expName     = 'E001-01'
size = [512 512  3];
path = "/DATASET/";
isMAT = false; 
splits = ["train", "test", "test"]
folders = ["RGB", "masks"]
savePredictionsFolder = '/temp/'


classNames = ["Class1","Class2", "Class3", "Class4"];
labelIDs   = [119 11 32; 111 74 0; 0 0 142; 0 60 100;]; % GT color per class

encoderDepth = 4;
filters = 32; 

batchsize   = 4;
epochs      = 120;
lr          = 1e-4;

useDataAugmentation = true;
L2Reg       = 0.0001; % 0.0001 (default)
gradientclipping = 1;


unet(expName, size, encoderDepth, filters, batchsize, epochs, useDataAugmentation, ...
    L2Reg, lr, gradientclipping, path, splits, folders, savePredictionsFolder, classNames, labelIDs, Inf, isMAT )


%%
clear
close all

expName     = 'E001-02'
size = [512 512  4];
path = "/DATASET/";
isMAT = true;
splits = ["train", "test", "test"]
folders = ["MAT", "masks"]
savePredictionsFolder = '/temp/'

classNames = ["Class1","Class2", "Class3", "Class4"];
labelIDs   = [119 11 32; 111 74 0; 0 0 142; 0 60 100;]; % GT color per class

encoderDepth = 4;
filters = 32; 

batchsize   = 8;
epochs      = 120;
lr          = 1e-3;

useDataAugmentation = true;
L2Reg       = 0.0001; % 0.0001 (default)
gradientclipping = 1;


unet(expName, size, encoderDepth, filters, batchsize, epochs, useDataAugmentation, ...
    L2Reg, lr, gradientclipping, path, splits, folders, savePredictionsFolder, classNames, labelIDs, Inf, isMAT )
