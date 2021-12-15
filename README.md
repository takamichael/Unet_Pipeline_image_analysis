# Unet_Pipeline_image_analysis
uses python and pytorch

To track an individual budding yeast cell overtime

This code is specifically made for my dataset which are 512x512 images collected at 63X. Three channels are taken into consideration: Brightfield, green, and red channel. 
Based on the first time, the pipeline will identify individual mothercells and track the single cell overtime. The code will NOT monitor any daughter cells.
The code has been tailored to use UNET and machine learning approaches, which take cell size, eccentricity, and area into consideration. 

Calculations peformed are used to determine flourescent or brightfield intensity on either all the merged zstacks or on individual zstacks. 


# File output Based on Summing the Z Stacks and then performing quantifications
'Files are outputed for the sum of all zstakcs and for individual zstacks. They are outputted as pickle file'

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

rfpmaxnorm_* - rfpdf/areamaskdf it is the max pixel of a single cell divided by the area of the cell for the RFP channel 

rfpsumnorm_* - rfpsumdf/areamaskdf it is the sum pixels of a single cell divded by the area of the cell for the RFP channel 

gfpmaxvoln_* - is the gfpdf/volumedf it is the max gfp value using the gaussian blur and divided by the volume of the cell 

gfpsumvoln_* - is the gfpsumdf/volumedf it is the sum GFP values using the gaussian blur and divided by the volume of the cell 

gfpmaxvolnori_* gfpdfori/volumedf it is the max gfp value divided by the volume of the cell 

gfpsumvolnori_* - is the gfpsumdfori/areamaskdf it is the sum GFP values and divided by the volume of the cell 

rfpmaxvolnm_* - rfpdf/volumedf it is the max pixel of a single cell divided by the volume of the cell for the RFP channel

rfpsumvolnrm_* - rfpsumdf/volumedf it is the sum pixels of a single cell divded by the volume of the cell for the RFP channel 

r2timedf_* - has some correlations between the individual cells gfp levels and time. Not a useful file 

avercorrdf_* - more correlations between the individual cells gfp levels. Not very useful 

backmeandfgfp_*  - mean background of the green channel, background is calculated after taking the sum of all zstacks 

backmediandfgfp_*  - median background of the gfp channel , background is calculated after taking the sum of all zstacks 

backmeandfrfp_*  - median background of the rfp channel , background is calculated after taking the sum of all zstacks 

backmediandfrfp_*  - median background of the rfp channel , background is calculated after taking the sum of all zstacks

gfpsumMeanback_* - gfpsumdf -backmeandf this is the  sum GFP values from gaussian blur subtracted from the back mean

gfpsumMedback_* - gfpsumdf -backmediandf this is the  sum GFP values from gaussian blur subtracted from the backmedian 

gfpsumMeanGMMAXback_* - gfpGMmaxdf-backmeandf subtracting backmean from mean max pixel after Gaus mixture model for the GFP channel when using a gaussian blur 

gfpsumMedGMMAXback_* - gfpGMmaxdf-backmediandf subtracting backmedian from mean max pixel after Gaus mixture model for the GFP channel when using a gaussian blur 

# File output Based on Individual Zstacks 

gfpsumzstackori_* - getting the sum of the pixels for every zstack from the original GFP channel every zstack pixel sum is taken per single cell

gfpGMmaxzstackori_*- getting the max mean gaussian mixture model for every zstack from the original gfp channel 

gfpGMminzstackori_*- getting the min mean gaussian mixture model for every zstack from the original gfp channel 

proteinareazstackori_* - getting the protein area of occupancy from the GFP channel 

backmedianlszstackgfp_*-getting the median background gfp for every zstack per single cell from the original GFP channel 

backmeanlszstackgfp_*- getting the mean background gfp for every zstack per single cell from the original GFP channel

nzstackgfpGMax_*- getting the SUM per zstack of the Max mean gaussian mixture model after subtracting the mean background per zstack. 

nmedzstackgfpGMax_*- getting the SUM per zstack of the Max mean gaussian mixture model after subtracting the median background per zstack. 

nzmaxstackgfpGMax_*- getting the SUM for the TOP 5 zstacks (based on abovemean) of the Max mean gaussian mixture model after subtracting the median background per zstack. 

bay5zstack_*- taking the top 5 zstacks (based on abovemean) it is the bayesian mixture model max sum  for the GFP channel 

abovemean_*- calculated mean pixels of a single cell greater than a particular threshold per zstack for the GFP channel

above5mean_* - calculated mean pixels of a single cell greater than a particular threshold for the top 5 zstacks for the GFP Channel

volume5gfp_* - calculate protein occupancy for the top 5 zstack (based on abovemean) of an individual cell

gfpsumzstackorinorm_*- the sum GFP pixels per zstack subtracted from the median background per zstack and divided by the volume of protein occupancy per zstack for each cell

volumezstack_* - calculate protein occupancy per zstack of an individual cell

gfpsum5zstackorinorm_* - samething as gfpsumzstackorinorm except that intead of using all the zstacks the top 5 zstacks are being used

rfpGMmaxzstackori_*-  getting the max mean gaussian mixture model for every zstack from the original RFP channel 

rfpGMminzstackori_*- getting the min mean gaussian mixture model for every zstack from the original RFP channel 

proteinareazstackorirfp_* -  getting the protein area of occupancy from the rfp channel 

backmedianlszstackrfp_*- getting the median background RFP for every zstack per single cell from the original RFP channel 

backmeanlszstackrfp_*-getting the mean background RFP for every zstack per single cell from the original RFP channel

nzstackrfpGMax_*-getting the SUM per zstack of the Max mean gaussian mixture model after subtracting the mean background per zstack. for RFP channel

nmedzstackrfpGMax_*-getting the SUM per zstack of the Max mean gaussian mixture model after subtracting the median background per zstack. for RFP channel

nzmaxstackrfpGMax_*- RFP getting the SUM for the TOP 5 zstacks (based on abovemean) of the Max mean gaussian mixture model after subtracting the median background per zstack. 

abovemeanrfp_*- calculated mean pixels of a single cell greater than a particular threshold per zstack for the RFP channel

above5meanrfp_*- calculated mean pixels of a single cell greater than a particular threshold for the top 5 zstacks for the RFP Channel

volume5rfp_*- calculate protein occupancy for the top 5 zstack (based on abovemean) of an individual cell for RFP channel 
 
rfpsumzstackorinorm_*-  the sum RFP pixels per zstack subtracted from the median background per zstack and divided by the volume of protein occupancy per zstack for each cell

rfpvolumezstack_*- calculate protein occupancy per zstack of an individual cell for RFP channel 

rfpsum5zstackorinorm_*- samething as rfpsumzstackorinorm except that intead of using all the zstacks the top 5 zstacks are being used

bfGMmaxzstackori_*-  getting the max mean gaussian mixture model for every zstack from the original brightfield channel 

bfGMminzstackori_*-  getting the min mean gaussian mixture model for every zstack from the original brightfield channel 

proteinareazstackoribf_*- getting the protein area of occupancy from the brightfield channel  channel 

backmedianlszstackbf_*-getting the median background RFP for every zstack per single cell from the original brightfield channel 
 
backmeanlszstackbf_*-getting the mean background brightfield for every zstack per single cell from the original brightfield channel

nzstackbfGMax_*- getting the SUM per zstack of the Max mean gaussian mixture model after subtracting the mean background per zstack. for brightfield channel

nmedzstackbfGMax_*-getting the SUM per zstack of the Max mean gaussian mixture model after subtracting the median background per zstack. for brightfield channel

nzmaxstackbfGMax_*-getting the SUM for the TOP 5 zstacks (based on abovemean) of the Max mean gaussian mixture model after subtracting the median background per zstack. BF

abovemeanbf_*-calculated mean pixels of a single cell greater than a particular threshold per zstack for the brightfield channel

above5meanbf_*-calculated mean pixels of a single cell greater than a particular threshold for the top 5 zstacks for the brightfield channel

volume5bf_*-calculate protein occupancy for the top 5 zstack (based on abovemean) of an individual cell for the brightfield channel

bfsumzstackorinorm_*-the sum BF pixels per zstack subtracted from the median background per zstack and divided by the volume of protein occupancy per zstack for each cell

bfvolumezstack_*- calculate protein occupancy for the top 5 zstack (based on abovemean) of an individual cell for Brightfield channel 

bfsum5zstackorinorm_*- samething as bfsumzstackorinorm except that intead of using all the zstacks the top 5 zstacks are being used
