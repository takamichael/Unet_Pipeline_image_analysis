# Unet_Pipeline_image_analysis
To track an individual budding yeast cell overtime

This code is specifically made for my dataset which are 512x512 images collected at 63X. Three channels are taken into consideration: Brightfield, green, and red channel. 
On the first time the pipeline will identify the individual mothercell and track the single cell overtime. 
The code has been tailored to use UNET and machinelearning approaches. 

Calculations peformed are used to determine flourescent or brightfield intensity on either all the merged zstacks or on individual zstacks. 
