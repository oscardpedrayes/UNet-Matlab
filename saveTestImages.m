function saveTestImages(net, imdsTest, pxdsTest, classNames, path, labelIDs, isMAT)


colormap = labelIDs ./ 255; %[0 0 0; 1 1 1];
transparency = 0.6;
outputImage = [];
%% Select the images
for i = [1, 20, 50, 80, 100, 120, 130]
    % Read image and mask
    imageFileName = char(imdsTest.Files(i));
    maskFileName = char(pxdsTest.Files(i));
    if isMAT==true % Read final_matrix from .MAT file
        load(imageFileName,'final_matrix');
        size(final_matrix)
        image = final_matrix(:,:,[1 2 3 4]);
        size(image)
    else
        image = imread(imageFileName);
    end
    mask = imread(maskFileName);
    % Predict
    prediction = semanticseg(image,net);
    if isMAT==true % If .MAT then convert image to uint8
        image= im2uint8(image(:,:,[1 2 3]));
        %mask= im2double(mask);
    end
    mask = labeloverlay(mask,prediction,'IncludedLabels',classNames, 'Colormap',colormap, 'Transparency',1);

    % Combine the image and the prediction
    predictionOverlay = labeloverlay(image,prediction,'IncludedLabels',classNames, 'Colormap',colormap, 'Transparency',transparency);
    prediction = labeloverlay(image,prediction,'IncludedLabels',classNames, 'Colormap',colormap, 'Transparency',0);
    % Combine the image and the ground truth
    maskOverlay = transparency.*image + (1-transparency).*mask;
    % Append images to the array

    if isMAT==true
        combination = [image mask prediction im2uint8(final_matrix(:,:,[4 4 4])) predictionOverlay];
    else
        combination = [image mask prediction maskOverlay predictionOverlay];
    end

    if  isempty(outputImage)
        outputImage = combination;
    else
        outputImage = [outputImage; combination];
    end
end


%% Save image
figure(1),clf(1) 
imshow(outputImage)
imwrite(outputImage, path);

end
