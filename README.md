# Unet_Pipeline_image_analysis
uses python and pytorch

To track an individual budding yeast cell overtime

This code is specifically made for my dataset which are 512x512 images collected at 63X. Three channels are taken into consideration: Brightfield, green, and red channel. 
On the first time, the pipeline will identify the individual mothercell and track the single cell overtime. The code will NOT monitor any daughter cells.
The code has been tailored to use UNET and machinelearning approaches. 

Calculations peformed are used to determine flourescent or brightfield intensity on either all the merged zstacks or on individual zstacks. 


###File Outputs:
Files are outputed for the sum of all zstakcs and for individual zstacks. They are outputted as pickle file

infodf_* - info file with the cell name, position, time, cell boundary, eccentricity, translation shift, and area
gfpGMmindf_* - This file contains the Gaussian mixture meanvalues of the smaller distribution from the GFP channel after a gaussian blur 
gfpGMmindfori_* - This file contains the Gaussian mixture mean values of the smaller distribution from the GFP channel  NO gaussian blur 
rfpGMmindf_* - This file contains the Gaussian mixture mean values of the smaller distribution from the RFP channel after a gaussian blur 
rfpdf_* - file contains the MAX pixel values of the single cells for the RFP channel 
rfpsumdf_* - the file contains the SUM pixel values of the single cells for the RFP channel 
rfpGMmaxdf_* -  the file contains the mean max pixel after Gaus mixture model for the RFP channel 
gfpdf_* - the file contains the max value of the pixel for each single cell when using a gaussian blur on the GFP channel
gfpdfori_* - the file contains the max value of the pixel for each single cell of the GFP channel
gfpsumdf_* - the file contains the sum of pixel values for each single cell of the GFP channel when using a gaussian blur
gfpsumdfori_* - the file contains the sum of pixel values for each single cell of the GFP channel 
gfpGMmaxdf_* the file contains the mean max pixel after Gaus mixture model for the GFP channel when using a gaussian blur
gfpGMmaxdfori_* - the file contains the mean max pixel after Gaus mixture model for the GFP channel 
proteinareadf_* the file contains the area of protein occupancy of the GFP channel. this is calculated by getting the sum of the total pixels occupied when using a threshold
areamaskdf_* - the file contains the area of a single cell overtime
radiusdf_* - the file contains the radius of each cell. This is calculated by using the areamaskdf file and assuming that the cell is always circular 
volumedf_* - the file contains the volume of each cell. It is calculated by using the areamaskdf file and assuming that the cell is a sphere shape
gfpmaxnorm_* - is the gfpdf/areamaskdf it is the max gfp values using the gaussian blur and divided by the area of the cell 
gfpsumnorm_* - is the gfpsumdf/areamaskdf it is the sum GFP values using the gaussian blur and divided by the area of the cell 
gfpmaxnormori_*- gfpdfori/areamaskdf it is the max gfp values and divided by the area of the cell 
gfpsumnormori_* - is the gfpsumdfori/areamaskdf it is the sum GFP values and divided by the area of the cell 



rfpmaxnorm.to_pickle(outfilename +str("rfpmaxnorm_" +str(date)+".pkl"))
rfpsumnorm.to_pickle(outfilename +str("rfpsumnorm_" +str(date)+".pkl"))
gfpmaxvoln.to_pickle(outfilename +str("gfpmaxvoln_" +str(date)+".pkl"))  
gfpsumvoln.to_pickle(outfilename +str("gfpsumvoln_" +str(date)+".pkl")) 
gfpmaxvolnori.to_pickle(outfilename +str("gfpmaxvolnori_" +str(date)+".pkl"))  
gfpsumvolnori.to_pickle(outfilename +str("gfpsumvolnori_" +str(date)+".pkl"))  
rfpmaxvolnm.to_pickle(outfilename +str("rfpmaxvolnm_" +str(date)+".pkl"))  
rfpsumvolnrm.to_pickle(outfilename +str("rfpsumvolnrm_" +str(date)+".pkl")) 
#straindic.to_pickle(outfilename +str("straindic_" +str(date)+".pkl"))
#countcellsdf.to_pickle(outfilename +str("countcellsdf_" +str(date)+".pkl"))

r2timedf.to_pickle(outfilename +str("r2timedf_" +str(date)+".pkl"))
avercorrdf.to_pickle(outfilename +str("avercorrdf_" +str(date)+".pkl"))


gfpsumMeanback.to_pickle(outfilename +str("gfpsumMeanback_" +str(date)+".pkl"))
gfpsumMedback.to_pickle(outfilename +str("gfpsumMedback_" +str(date)+".pkl"))

gfpsumMeanGMMAXback.to_pickle(outfilename +str("gfpsumMeanGMMAXback_" +str(date)+".pkl"))
gfpsumMedGMMAXback.to_pickle(outfilename +str("gfpsumMedGMMAXback_" +str(date)+".pkl")) 


backmeandf.to_pickle(outfilename +str("backmeandf_" +str(date)+".pkl")) 
backmediandf.to_pickle(outfilename +str("backmediandf_" +str(date)+".pkl")) 

backmeandfgfp_*  - mean background of the green channel, background is calculated after taking the sum of all zstacks 
backmediandfgfp_*  - median background of the gfp channel , background is calculated after taking the sum of all zstacks 
backmeandfrfp_*  - median background of the rfp channel , background is calculated after taking the sum of all zstacks 
backmediandfrfp_*  - median background of the rfp channel , background is calculated after taking the sum of all zstacks
