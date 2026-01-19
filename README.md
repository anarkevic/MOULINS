Mosaic Utility and Large Dataset Integration for SERAC (MOULINS)

This workbook will guide you through the process of using laser altimetry to check the accuracy of or correct a digital elevation model (DEM). Please ensure that you download the modules "moulinsWorkbookModule050525.py" and "pyMTDForMoulinsModule050525.py" and place them in the same directory as the Jupyter notebook file "moulinsWorkbook050525.ipynb"

The following packages are required for MOULINS, and should be installed in your Python environment:
numpy
shapely
pyproj
PIL
rasterio
scipy
matplotlib
netCDF4

The following packages are necessary only for floating ice corrections:
inspect
re

Altimetry is made available by the University at Buffalo Remote Sensing Lab, but users must provide their own DEMs, and they must be able to specify the EPSG code for the projection used by that DEM. The DEM must also be vertically referenced to the WGS-84 Ellipsoid and in the GeoTIFF format. Contact the creator of your DEM if you don't know the projection or vertical datum or file format. You must rename any DEMs you supply so that MOULINS will recognize them, using the DEM date in MMDDYY format and a string of characters that will identify the DEM in the generated time series, followed by "_dem.tif" The filename should look something like "32416AST_dem.tif" or "100121WV_dem.tif" The code can by any number of characters, but the format of the time series files may be less readable if you use more than three characters, and the filename should have no numerals that are not part of the date.

*****************************************************************

(Block 1) Set-up Step

Edit the variable values in the code below to suit your needs.
useCurrentDirectory: (True/False) Set this to True to use your current directory as the working directory, or False to specify a directory
  ex: useCurrentDirectory = True
  or: useCurrentDirectory = False
directoryToUse: (''/file path) If you entered "False" for useCurrentDirectory, please supply the full directory of the file path you would like to use as the MOULINS working directory. The path should be in quotes, and all slashes must be doubled. If you chose True for useCurrentDirectory, this should be set to empty quotes.
  ex: directoryToUse = 'C:\Users\ACoolGeophysicist\example_folder\moulins'
  or: directoryToUse = ''
prefix: (string of characters) In case you use MOULINS multiple times in the same directory, a file prefix is utilized so that the user can differentiate between different runs. Use any string you wish, in quotes.
  ex: directoryToUse = 'example61323'
referenceDate (integer) Choose a reference date for the SERAC time series. Changes will be calculated relative to this date. Please use MMDDYY format. (You may omit the leading zero for pre-October months.)
  ex: referenceDate = 91518

Below this is code to set the current directory and import the MOULINS python package.

Always run this block before starting to a session to ensure your directory and basic attributes are initialized, and the MOULINS package is loaded.

*****************************************************************

(Block 2) Pre-Step 1: File Architecture Generation

MOULINS uses a specific file architecture to organize input and output files. Run this block to generate the architecture within the previously specified directory. Once this block has been run, it does not need to be run again unless you wish to change the MOULINS working directory. However, using it again when architecture already exists will not result in the deletion of pre-existing data; in that case this block will simply do nothing.
After generating the file architecture please ensure that your DEM is in the "DEM" folder, the appropriate altimetry files are in the "Altimetry" folder, the file GRE_IceSheet_IMBIE2_v1.exp is in the IceSheetBoundary Folder, and the gsfc_fdm_v1_2_1_gris_Dec22.nc file is in the ClimateModel folder.
Do not run this block until you have run Block 1.

*****************************************************************

(Block 3) Pre-Step 2: Shapefile Generation

To reduce processing time, MOULINS creates shapefiles for all the provided DEMs that define the convex hull of the contained data points. This allows for faster retrieval of altimetry points within the domain of each DEM. Note that the shapefile is not actually in ArcGIS format, but is a .exp text file with the coordinates of each vertex. Depending on DEM size, this could take over 1 minute per DEM. Upon completion you will see a map with the outline of the Greenland Ice Sheet in black and the combined outline of the input DEMs in red, so you can verify that the spacial references are correct.

*****************************************************************

(Block 4) Pre-Step 3: Altimetry Subsetting

Altimetry data point that fall within the bounds of the provided DEMs are extracted and stored in a seperate temporary file within the MOULINS architecture. This negates the need to perform such a search during each step of MOULINS that requires reading altimetry. Only data relevant to MOULINS is extracted and stored in this file.
In this step you must also specify the EPSG code for the DEM projection. Some examples:

For Latitude and Longitude:
  dem_projection = 4326

For UTM N (## = UTM zone):
  dem_projection = 426##

For NSIDC North Polar Stereographic:
  dem_projection = 3413 

*****************************************************************

(Block 5) Step 1: Control Point Generation

This step will review the altimetry and generate optimized control point locations that meet minimum criteria. The output folder will contain a map of the control point locations and altimetry distribution for each point. Edit the values in the code block below until you get a satisfactory density and distribution of control points. The variables are as follows:

Date radius: only altimetry within this number of years of the input DEM dates will be used.
 ex: date_radius = 25

Buffer time: the minimum amount of time, in years, in the altimetry record before and after the DEM date. (Ex: if the input DEM has a date of 9/1/2020 and the buffer time is set to 0.25 (~3 months), all control points will have altimetry records that begin no later than the the start of June 2020, and end no earlier than the start of December 2020.)
 ex: buffer_time = 0.25

Epoch buffer: the minimum number of unique altimetry dates in the record before and after the DEM date. (This should not be lower than 2!)
 ex: epoch_buffer = 2

Maximum time gap: the maximum allowed gap in the altimetry record in which the DEM date resides. (Ex: if the input DEM has a date of 9/1/2020 and the maximum time gap is set to 3, a control point with a gap in the altimetry record between 1/1/2016 and 1/1/2021 will be rejected, but a control point with a gap in the record from 1/1/2019 to 1/1/2021 will be accepted. 3 years is strongly recommended for this value.
 ex: max_time_gap = 3

*****************************************************************

(Block 6) Step 2: Altimetry extraction by surface patch

For each control point, this step locates all altimetry data within its surface patch (i.e., a 1 km square centered on the control point.) This step is equivalent to SERAC Program 1, and produces individual text files for each control point for the sake of compatibility and user review.
The output files have the following format:
Filename:
  [prefix] + 'Patch' + [control point id] + '.txt'
Header Line:
  Control point id    Number of altimetry points    Center x coordinate    Center y coordinate
Body lines:
  Sensor ID    Date (MMDDYY)    Date (decimal year)    X coordinate    Y coordinate    Elevation

*****************************************************************

(Block 7) Step 3 (Optional): Altimetry Time Series Generation

This step generates SERAC time series using only altimetry. This is not necessary, but is provided as an option for comparison. If the shape parameters or error estimates vary wildly between the altimetry time series and final time series including DEMs, this can indicate that the DEM and altimetry are poorly matched and you should review the input data. The specifications of the output files are described below in (Block 11) Step 6.

*****************************************************************

(Block 8) Step 4: DEM Extraction by Surface Patch

For each of the generated control points, this step will find all the pixels from each DEM that fall within a 1 km square centered on that control points, and save the elevations in a series of output text files. These files contain the same information as the altimetry files generated by (Block 7) Step 2.

*****************************************************************

(Block 9) Step 5: DEM Surface Patch Analysis

This step checks to make sure that each DEM provides a stable surface shape solution at the location of each control point within its bounds. If the standard deviation of the residuals from the least-squares fitting exceeds 10, that control point will not be used for that DEM (and vice-versa). The output files contain the following information for each DEM:

timeEpoch: The realtive chronological order of the DEMs
date: The date of each DEM (in MMDDYY format)
#points: The number of pixels used for the fitting from each DEM
averageZ: The mean elevation of all pixels used for each DEM
sigma: The standard deviation of residuals from the least-squares fitting.
shapePar1 and shapePar2: The first two coefficients from the fitted polynomial surface.
absoluteZ: The elevation of the fitted polynomial surface at the center of the surface patch.
flag: The flag which determines whether each DEM will be used in the final time series for each control point.

*****************************************************************

(Block 10) Step 6: Combined Time Series Generation

This step generates an elevation time series from the combination of altimetry and DEM data, using least-squares fitting. This produces two files, the "stats" file and the "dhdt" file. The stats file keeps a record of information from the fitting that is not necessary for the time series, but may be helpful to some users. The stats file is a text document with information for each control point, successively. For each control point, the following information is presented:

CPID: the control point identifier assigned in Step 1
height at centroid: the elevation of the centroid of all the input data points
#points: the number of input data points
#blunders: the number of data points removed in the final iteration due to residues exceeding a 3 sigma tolerance
kappa: the condition number of the fitting. This represents the solution's sensitivity to error in the input. The magnitude may vary depending on the amounts and types of inputs, and should be viewed relatively. I.e., a control point with a much higher kappa than the rest is "ill-conditioned" and should be regarded cautiously.
iteration: the following lines are divided into columns, with each column representing a fitting iteration
sigma_0: the standard deviation of the fitting for each iteration
a01-a09: the shape parameters of the fitted polynomial surface
h00-hn: the temporally-fitted centroid height for each unique date in the time series

The dhdt file contains the actual SERAC time series for each control point, successively. At the top of the document, below the two header lines, is the number of control points for which time series were determined. Beneath this is the data for each control point. The first line contains the following information:

cpid: as above
numItems: the number of unique dates in the time series
x, y, z: the centroid coordinates
kappa, sigma0: as above
referenceTime: the user-specified reference date

Then, for each unique date in the time series, the following information is given:

mission: the mission code (generally "ATM," "ICE," "IC2" for altimetry, and user-specified tags for DEMs)
decDate: the date, specified as a fractional year (e.g., 2014.3123)
MMDDYY: the date in mm/dd/yy format (e.g., 42414 for the previous example)
months: the time difference with respect to the reference date in fractional months
relativeZ: the elevation difference relative to the estimate centroid elevation on the chosen reference date
absoluteZ: the absolute centroid elevation (relative to the WGS84 datum)
sigma: the standard deviation of the residues of only the points from this date
numPoints: the number of input elevation data points (after the removal of blunders)
blunders: the number of data points removed from the final iteration
demFlag: a marker for DEM dates, to facilitate later steps

*****************************************************************

(Block 11) Step 7: Time Series Curve Fitting

This step fits a curve to the time series from the previous step so that the "true" elevation of DEMs at each control point can be interpolated. The algorithm uses a penalized spline fit to estimate the DEM corrections, as well as a polynomial fit for visual comparison with the spline fit. Using the climate model file supplied with the code, this step will also remove seasonal effects from the time series, improving the fitting, and protecting any seasonal signals in the DEM itself from being removed upon correction. If the climateModel variable to set equal to True, the effects of seasonal firn density changes will be removed from the time series. If mosaicMode is set equal to True, this will only be applied to the altimetry data, such that firn-based variability will be removed from the DEM during correction. If mosaicMode is set to True, make sure to set an appropriate date for the correction (refDate). This is the date all the DEMs will be corrected to (only if mosaicMode is true). If you do not have the file gsfc_fdm_v1_2_1_gris_Dec22.nc in the ClimateModel folder, ensure that this variable is set equal to False.
This step produces a text file with the following format for each input DEM:

CPID: the control point identifier (for each control point used for this DEM)
x: the x coordinate of that control point
y: the y coordinate of that control point
corrALPS: the correction vector for the penalized spline fit, i.e., the amount of vertical correction needed to make the DEM agree with the time series at that control point
resALPS: the standard deviation of residuals from the penalized spline fitting
corrPoly: the polynomial correction vector
resPoly: the standard deviation of residuals from the polynomial fitting
flag: This flag determines whether the control point will be used in the DEM correction ('0') or not ('1'). This flag is automatically set for any control point with a spline fit correction vector that is more than three standard deviations outside from the average correction vector of all control points for that DEM. The user can also manually flag points by editing this textfile.

This step also generates a PDF, with TimeSeriesPlots in the name, of visualizations for the time series at each control point. These graphs include the following elements:

top caption: the control point ID
x axis: time, in calendar years
y axis: elevation change of surface element centroid relative to the user-given reference date
red dots: points in the time series that were determined from altimetry
blue dots: points in the time series that were determined from DEMs
green line: the polynomial best-fit line
green shading: 95% confidence interval for the polynomial fit
black line: the best-fit spline
gray shading: 95% confidence interval for the spline fit
lower left caption: the dates of all the DEMs that use that control point

Finally, there is a second PDF, with DEMPlots in the name. This consists of an outline of each DEM, and a scatterplot of the control points used, which are colored based on their correction vector. This can be used to visually examine the pattern of error estimates in the DEM.

*****************************************************************

(Block 12) Step 8: Correction Surface Fitting
Note: if you are using this pipeline to check DEMs for accuracy only, you do not need to proceed beyond Step 7.

This step uses the previously generated control vectors to compute a best-fit correction surface. This surface represents the amount of correction to be applied to each area of the DEM. The algorithm will calculate 2D polynomial corrections for 1st-3rd order polynomials and decide which is best, based on the mean improvement at each control point, and the degree of warping (which is determined by comparing the mean value of the entire surface to the mean value at the control points). The polynomial degree determination is currently still in testing, so users are advised to check the visualizations and edit the polynomial degree of the best-fit surface as they see fit. For each DEM, this step will generate a text file summarizing the details of the surface fitting, which contains the following information:

centroid: the centroid coordinates of the surface patch, i.e., the control point coordinates
Deg #: the fitting for each polynomial degree is described sequentially, separated by this header
Iteration: control points with residuals more than three times the standard deviation of all residuals will be excluded, if any, and the fitting repeated up to 1 additional time; this header signifies which iteration each column represents
A0...A9: the polynomial shape paramaters (3 for 1st degree, 6 for 2nd, 9 for 3rd)
control point ids: for each control point used in that DEM, the residual difference between the time series elevation on the DEM date and the corrected DEM are given for each iteration
Mean control vector is: the mean of the input correction vectors
RMS control vector is: the root mean square magnitude of the correction vectors
RMS residual is: the root mean square value of the residuals shown above for the final iteration
RMS correction value is: the root mean square value of the correction surface at a random sampling of locations
Improvement is: the difference between the RMS control vector and RMS residual
Fit ratio is: the ratio of the RMS correction value to the RMS value of the correction surface at the control points (which is not the same as the RMS correction vector!); since underfitting and overfitting are both bad, the ratio is arranged in whichever way give a number less than 1, so that a lower fit ratio is always worse
Decision weight is: fit ratio times the improvement divided by the RMS correction vector (work in progress)
Control points used: at the bottom of the document, after the Degree 3 summary, this gives the number of control points used in the fitting for the selected polynomial degree
Use degree: At the very bottom of the document, this gives the polynomial degree that will be used for the correction in Step 9. The user may modify this to any of the available degrees as they see fit, or to "NA" if the correction for this DEM is deemed to be unusable.

The output folder also includes three PDFs per DEM, which include visualizations of the control vectors and the correction surface, with each representing a different polynomial degree. Each file contains nine figures, representing the same surface in 3 dimensions from nine different viewing angles. The blue mesh in these images represents the correction surface. The black line superimposed onto the mesh is the boundary of the DEM. And the red dots represent the control vectors used in the surface fitting.

Special note: For this version, the recommneded Degree will always be 1, since only Worldview DEMs are being used, and other degrees are consistent with the typical Worldview error patterns.

*****************************************************************

(Block 12) Step 9: Surface Correction

This step simply applies the calculated correction to each of the input DEMs. It reads the polynomial coefficients for the degree specified in the Step 8 output textfiles, and adds the resultant surface to the DEM values to create the final corrected DEMs.

*****************************************************************

(Block 13) Step 10a: Floating Ice Corrections (Float Mask Generation, Altimetry Correction)

This block will identify on which pixels the ice sheet is floating, and which control points are on floating ice. Please ensure that you have the latest version of the BedMachine bed and geoid geotiff files in the BedMachine folder. In the block below, enter the desired values for the density of water and ice that will be used to determine where ice is floating based on an assumption of neutral buoyancy. Also once more please specify the projection of the input DEMs via ESRI code.

The main output folder contains masks showing where the floating ice is located within each DEM, and a text file listing each DEM that was determined to contain floating ice.

*****************************************************************

(Block 14) Step 10b: Floating Ice Corrections (Recompute time series, correction surface, and DEM correction)

This block will repeat earlier blocks of code to generate time series, correction surfaces, and corrected DEMs for all DEMs determined to have floating ice. These outputs can be found in appropirately labeled subfolders within the main step10/[prefix]floating folder. In the block below, indicate whether the firn density model is being applied to this run (i.e., climateMode = True if you desire the dynamic component of elevation change), and mosaicMode determines whether the model is applied to DEM values (i.e., mosaicMode = True to apply it and mitigate seasonal effects to improve mosaicking DEMs that are several months apart.
