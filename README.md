# Unet_Pipeline_image_analysis
uses python and pytorch

To track an individual budding yeast cell overtime

This code is specifically made for my dataset which are 512x512 images collected at 63X. Three channels are taken into consideration: Brightfield, green, and red channel. 
On the first time, the pipeline will identify the individual mothercell and track the single cell overtime. The code will NOT monitor any daughter cells.
The code has been tailored to use UNET and machinelearning approaches. 

Calculations peformed are used to determine flourescent or brightfield intensity on either all the merged zstacks or on individual zstacks. 


File Outputs:
Files are outputed for the sum of all zstakcs and for individual zstacks. They are outputted as pickle file



backmeandfgfp_*  - mean background of the green channel, background is calculated after taking the sum of all zstacks 
backmediandfgfp_*  - median background of the gfp channel , background is calculated after taking the sum of all zstacks 
backmeandfrfp_*  - median background of the rfp channel , background is calculated after taking the sum of all zstacks 
backmediandfrfp_*  - median background of the rfp channel , background is calculated after taking the sum of all zstacks 

 
infodf_* - info file with the cell name, position, time, cell boundary, eccentricity, translation shift, and area
gfpGMmindf_*

gfpGMmindfori.to_pickle(outfilename +str("gfpGMmindfori_" +str(date)+".pkl"))
rfpGMmindf.to_pickle(outfilename +str("rfpGMmindf_" +str(date)+".pkl"))

rfpdf.to_pickle(outfilename +str("rfpdf_" +str(date)+".pkl"))
rfpsumdf.to_pickle(outfilename +str("rfpsumdf_" +str(date)+".pkl"))
rfpGMmaxdf.to_pickle(outfilename +str("rfpGMmaxdf_" +str(date)+".pkl"))
gfpdf.to_pickle(outfilename +str("gfpdf_" +str(date)+".pkl"))
gfpdfori.to_pickle(outfilename +str("gfpdfori_" +str(date)+".pkl"))
gfpsumdf.to_pickle(outfilename +str("gfpsumdf_" +str(date)+".pkl"))
gfpsumdfori.to_pickle(outfilename +str("gfpsumdfori_" +str(date)+".pkl"))
gfpGMmaxdf.to_pickle(outfilename +str("gfpGMmaxdf_" +str(date)+".pkl"))
gfpGMmaxdfori.to_pickle(outfilename +str("gfpGMmaxdfori_" +str(date)+".pkl"))
proteinareadf.to_pickle(outfilename +str("proteinareadf_" +str(date)+".pkl"))
areamaskdf.to_pickle(outfilename +str("areamaskdf_" +str(date)+".pkl"))
radiusdf.to_pickle(outfilename +str("radiusdf_" +str(date)+".pkl"))
volumedf.to_pickle(outfilename +str("volumedf_" +str(date)+".pkl"))
gfpmaxnorm.to_pickle(outfilename +str("gfpmaxnorm_" +str(date)+".pkl"))
gfpsumnorm.to_pickle(outfilename +str("gfpsumnorm_" +str(date)+".pkl"))
gfpmaxnormori.to_pickle(outfilename +str("gfpmaxnormori_" +str(date)+".pkl"))

gfpsumnormori.to_pickle(outfilename +str("gfpsumnormori_" +str(date)+".pkl"))
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
