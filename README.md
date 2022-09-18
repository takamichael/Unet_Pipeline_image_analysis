# UNET_lapvar_Pipeline_Image_analysis  
Languages used: Python and Pytorch

To track an individual budding yeast cell overtime

This code is specifically made for my time series dataset which are 512x512 images collected at 63X. Three channels are taken into consideration: Brightfield, green, and red channel. 
Based on the first time, the pipeline will identify individual mothercells and track the single cell overtime. The code will NOT monitor any daughter cells.
The code has been tailored to use UNET and machine learning approaches, which take cell size, eccentricity, and area into consideration. 

Calculations peformed are used to determine flourescent or brightfield intensity on either all the merged zstacks or on individual zstacks. 

NoteBooks:


UNET_lapvar_Pipeline_Image_analysis.ipynb- Current pipeline used. Depends on two UNET models with Watersheding and Machine learning approaches to track a single cell

UNET_Allcell_notebook.ipynb - Notebook used to train the UNET Model to detect all the cells in a 512x 512 image

Unet_Pipeline_image_analysis.ipynb - is the old UNET pipeline and depended on a UNET that tracked just the mothercell 

UNET_Mothercell_notebookGitHubV.ipynb - Notebook used to trian the UNET model to detect ONLY the single mother cells. This UNET model was used with the Unet_Pipeline_image_analysis.ipynb notebook

requirements_Unet_lapvar_pipeline_6_10_2022.txt - is the requirment textfile which responds to the packages used for the UNET_lapvar_Pipeline_Image_analysis.ipynb


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

gfpvolzstacknorm_* - Sum of protein occupancy per zstack for each cell subtracted from the median background per zstack and divided by the volume of protein occupancy per zstack for each cell 

gfpsumzstackmedarea_*- sum of all the protein in every zstack with the median background divided by the cell area

gfpGMmaxzstacktop4_* - Per zstack peform gaussian filter of 8 next. From the original mothercell determine the mean pixels >= the gausssian mixture model from the blured image. Note the Median background is NOT subracted from this value 

gfpGMminzstacktop4_*- Per zstack peform gaussian filter of 8 next. From the original mothercell determine the mean pixels <= the gausssian mixture model from the blured image. Note the Median background is NOT subracted from this value 

gfpmeanzstack_*- Mean of all the protein of an individual mother cell in every zstack subtracting the median background and then summing the values 

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

rfpvolzstacknorm_* - Sum of protein occupancy per zstack for each cell subtracted from the median background per zstack and divided by the volume of protein occupancy per zstack for each cell 

rfpvolumezstack_*- calculate protein occupancy per zstack of an individual cell for RFP channel 

rfpsumzstackmedarea_*- sum of all the protein in every zstack - the median background divided by the cell area

rfpmeanzstack_*- Mean of all the protein of an individual mother cell in every zstack subtracting the median background and then summing the values 

rfpGMmaxzstacktop4_* - Per zstack peform gaussian filter of 8 next. From the original mothercell determine the mean pixels >= the 
gausssian mixture model from the blured image. Note the Median background is NOT subracted from this value 

rfpGMminzstacktop4_*- Per zstack peform gaussian filter of 8 next. From the original mothercell determine the mean pixels <= the gausssian mixture model from the blured image. Note the Median background is NOT subracted from this value 

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

bfvolzstacknorm_* - Sum of protein occupancy per zstack for each cell subtracted from the median background per zstack and divided by the volume of protein occupancy per zstack for each cell

bfsumzstackmedarea_*- sum of all the protein in every zstack - the median background divided by the cell area


bfGMmaxzstacktop4_* - Per zstack  peform gaussian filter of 8 next. From the original mothercell determine the mean pixels >= the 
gausssian mixture model from the blured image. Note the Median background is NOT subracted from this value 

bfGMminzstacktop4_*- Per zstack peform gaussian filter of 8 next. From the original mothercell determine the mean pixels <= the gausssian mixture model from the blured image. Note the Median background is NOT subracted from this value 

bfmeanzstack_*- Mean of all the protein of an individual mother cell in every zstack subtracting the median background and then summing the values

bfvolumezstack_*- calculate protein occupancy for the top 5 zstack (based on abovemean) of an individual cell for Brightfield channel 

bfsum5zstackorinorm_*- samething as bfsumzstackorinorm except that intead of using all the zstacks the top 5 zstacks are being used

## Files made by Aggregating zstacks together and peforming a Gaussian Mixture Model. To obtain proper demixing, a gausian filter of 3.5 is applied to the original image. Afterwhich, both the gaussian blurred image and mask of an individual mother is fed to the model. Two thresholds are calculated, the Gaussian mixture Model Max threshold is kept. The GMM area, mean, and sum  is calculated by using the ORIGINAL image and collecting pixels greater or less than the GMM max Threshold. I then turned everything outside of the mother cell into 0 before collecting the coordinates, area, and values. Any file with *norm indicates that they were normalized by subtracting the median background, this is calculated by obtaining the background of individual zstacks and summing it together per cell per individual time, 

gfpGMmaxareadf_* - counting the sum of boolan values greater than or equal to the GMM threshold. Keep in mind values are from AGGREGATED zstacks. For GFP channel

gfpGMminareadf_* - counting the sum of boolan values less than the GMM threshold. Keep in mind values are from AGGREGATED zstacks. For GFP channel

gfpGMmaxsumdfori_* -  Sum of pixels, per mother cell,greater than or equal to the GMM Threshold . Keep in mind values are from AGGREGATED zstacks.  For GFP channel

gfpGMminsumdfori_* - Sum of pixels, per mother cell,less than the GMM Threshold. Keep in mind values are from AGGREGATED zstacks. For GFP channel

gfpGMmaxdfori_* - The mean of the pixels, per mother cell, greater than or equal to the GMM Threshold. The intensity of the flourescent of the region. For GFP channel

gfpGMmindfori_* - getting the mean of the pixels, per mother cell, less than to the GMM Threshold. Gives intensity of the flourescent of the region. For GFP channel

gfpGMmaxcoor_* - the coordinates of the region that was greater than or equal to the GMM threshold. Keep in mind values are from AGGREGATED zstacks. For GFP channel

gfpGMmincoor_* - the coordinates of the region that was less than to the GMM threshold. Keep in mind values are from AGGREGATED zstacks. For GFP channel

gfpGMmaxsumdforinorm_* - gfpGMmaxsumdfori_* values are subtracted from the median background calculated per zstack. For GFP channel 

gfpGMminsumdforinorm_* - gfpGMminsumdfori_* values are subtracted from the median background calculated per zstack. For GFP channel 

gfpGMmaxdforinorm_* - gfpGMmaxdfori_* values are subtracted from the median background calculated per zstack. For GFP channel 

gfpGMmindforinorm_* - gfpGMmindfori_* values are subtracted from the median background calculated per zstack. For GFP channel 


rfpGMmaxareadf_* - counting the sum of boolan values greater than or equal to the GMM threshold. Keep in mind values are from AGGREGATED zstacks. For RFP channel

rfpGMminareadf_* - counting the sum of boolan values less than the GMM threshold. Keep in mind values are from AGGREGATED zstacks. For RFP channel

rfpGMmaxsumdfori_* -  Sum of pixels, per mother cell,greater than or equal to the GMM Threshold . Keep in mind values are from AGGREGATED zstacks.  For RFP channel

rfpGMminsumdfori_* - Sum of pixels, per mother cell,less than the GMM Threshold. Keep in mind values are from AGGREGATED zstacks. For RFP channel

rfpGMmaxdfori_* - The mean of the pixels, per mother cell, greater than or equal to the GMM Threshold. The intensity of the flourescent of the region. For RFP channel

rfpGMmindfori_* - getting the mean of the pixels, per mother cell, less than to the GMM Threshold. Gives intensity of the flourescent of the region. For RFP channel

rfpGMmaxcoor_* - the coordinates of the region that was greater than or equal to the GMM threshold. Keep in mind values are from AGGREGATED zstacks. For RFP channel

rfpGMmincoor_* - the coordinates of the region that was less than to the GMM threshold. Keep in mind values are from AGGREGATED zstacks. For RFP channel

rfpGMmaxsumdforinorm_* - rfpGMmaxsumdfori_* values are subtracted from the median background calculated per zstack. For RFP channel 

rfpGMminsumdforinorm_* - rfpGMminsumdfori_* values are subtracted from the median background calculated per zstack. For RFP channel 

rfpGMmaxdforinorm_* - rfpGMmaxdfori_* values are subtracted from the median background calculated per zstack. For RFP channel 

rfpGMmindforinorm_* - rfpGMmindfori_* values are subtracted from the median background calculated per zstack. For RFP channel 



bfGMmaxareadf_* - counting the sum of boolan values greater than or equal to the GMM threshold. Keep in mind values are from AGGREGATED zstacks. For BF channel

bfGMminareadf_* - counting the sum of boolan values less than the GMM threshold. Keep in mind values are from AGGREGATED zstacks. For BF channel

bfGMmaxsumdfori_* -  Sum of pixels, per mother cell,greater than or equal to the GMM Threshold . Keep in mind values are from AGGREGATED zstacks.  For BF channel

bfGMminsumdfori_* - Sum of pixels, per mother cell,less than the GMM Threshold. Keep in mind values are from AGGREGATED zstacks. For BF channel

bfGMmaxdfori_* - The mean of the pixels, per mother cell, greater than or equal to the GMM Threshold. The intensity of the flourescent of the region. For BF channel

bfGMmindfori_* - getting the mean of the pixels, per mother cell, less than to the GMM Threshold. Gives intensity of the flourescent of the region. For BF channel

bfGMmaxcoor_* - the coordinates of the region that was greater than or equal to the GMM threshold. Keep in mind values are from AGGREGATED zstacks. For BF channel

bfGMmincoor_* - the coordinates of the region that was less than to the GMM threshold. Keep in mind values are from AGGREGATED zstacks. For BF channel

bfGMmaxsumdforinorm_* - bfGMmaxsumdfori_* values are subtracted from the median background calculated per zstack. For BF channel 

bfGMminsumdforinorm_* - bfGMminsumdfori_* values are subtracted from the median background calculated per zstack. For BF channel 

bfGMmaxdforinorm_* - bfGMmaxdfori_* values are subtracted from the median background calculated per zstack. For BF channel 

bfGMmindforinorm_* - bfGMmindfori_* values are subtracted from the median background calculated per zstack. For BF channel 

# Files based on aggregating zstacks calculated using the Gaussian Mixture model coordinates from another channel The file should appear as the first * flourescent marker that is being quantified and the second * which is used to make the mask. Example: the mask is made using the 'rfpGMmaxcoor' and then getting the protein sum or mean of another channel like in the GFP or BF thus file would be 'gfphybrfpGmaxsumnorm_*. the values come from aggregating the images together'

*hyb*Gmaxsumnorm_*- GMM max of reference channel for the pixel sum of another channel, which is then subtracted from the sum zstack backmedianlszstack 

*hyb*Gminsumnorm_*- GMM min of reference channel for the pixel sum of another channel, which is then subtracted from the sum zstack backmedianlszstack 

*hyb*Gmaxnorm_*-  GMM max of  reference channel used for the pixel Mean of another channel, which is then subtracted from the sum zstack backmedianlszstack 

*hyb*Gminnorm_* - GMM min of  reference channel used for the pixel Mean of another channel, which is then subtracted from the sum zstack backmedianlszstack  

*hyb*intensity_* - *hyb*Gminnorm_* and *hyb*Gmaxnorm_* summed which then gives you the intensity as marker is normalized for area of pixels and background 

*hyb*Gmaxsum_* - GMM max of  reference channel used for the pixel sum of another channel per zstack 

*hyb*Gminsum_*- GMM min of  reference channel used for the pixel sum of another channel per zstack 

*hyb*Gmaxmean_*- GMM max of  reference channel used for the pixel mean of another channel per zstack 

*hyb*Gminmean_*-GMM min of  reference channel used for the pixel mean of another channel per zstack 
