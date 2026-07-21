# Continuity approaches for mass balance Intercomparison eXercise (ContinuIX): WP2 & WP3 protocol

# 

# What is ContinuIX?

The overall goal of this IACS working group (WG) is bringing together the research community that is developing continuity inversion and mass-conservation methodologies. These approaches derive glacier surface mass balance directly from glacier dynamics \- using distributed surface velocity, ice thickness and elevation change products \- as an alternative or complement to traditional climate-driven models. We use the best data there is out there and test which approaches can ‘reliably’ infer SMB quantities. We aim to answer the following research questions based on high-quality data from reference glaciers:

- What approaches are out there and how do they perform?  
- Can these models reproduce meaningful values?   
- How valid are uncertainty estimates?  
- How do/Can approaches cope with real data and can we quantify the effect on performance?


The WG is organized in three work packages (WPs) as follows:

**WP1.** Led by Maaike Izeboud and Albin Wells. Compiling a benchmark collection of high-quality contemporaneous ice thickness, surface elevation changes, ice surface velocity, and in situ surface mass balance data for several glaciers around the world as well as synthetic cases, along with uncertainty estimates and metadata for each dataset. 

**WP2.** Led by Johannes Fürst and Marin Kneib. Conducting a structured intercomparison experiment to understand the differences between approaches and the importance of key methodological aspects, using the best possible datasets (compiled in WP1). This experiment has a twofold objective. First it will evaluate the model performances, assess individual uncertainty estimates as well as utility of the collective SMB spread. Second it should serve as a reference to support/guide future methodological developments. 

**WP3.** Led by Evan Miles and Victor Devaux-Chupin. Conducting a structured intercomparison experiment to understand the effects of reduced qualities of input data on the approaches’ results. Key aspects to be investigated include the impact of resolution and errors or gaps in input datasets, by using degraded datasets from WP1. This experiment will serve as a reference to guide understanding of the data input requirements for established methods, and allow determining which inputs are most impactful for continuity inversion results.

# Experimental setup

## Data

### Synthetic cases

- One glacier run with Elmer/Ice simulation and a 10-year perturbation after equilibration (+/-ELA & East-West gradient with sampling of uncertainty; 10 experiments)  
- One basic geometry run with Icepack until equilibrated state (with uncertainty sampling; 10 experiments)

### Real-world cases

- 6 glaciers compiled from WP1

### 

### Input data format

Participants will only get input data & uncertainties homogenised on a georeferenced grid:

- DHDT: elevation change (unit: m yr-1 i.e.)  
- VX: x-component of velocity unit: m yr-1)  
- VY: y-component of velocity (unit: m yr-1)  
- THK: ice thickness (unit: m i.e.)  
- DEM: surface topography (unit: m a.s.l.)  
- BED: basal topography (unit: m a.s.l.)  
- ELEVBINS: discretization of surface topography in bins of 50 m elevation (unit: m a.s.l.) (to use if you are working with a flux-gate approach)  
- ICEMASK: labelled glacierized area (unit: yr; each pixel shows the last year where it was part of the glacier).   
- BASINMASK: only relevant for ice cap with multiple basins (unit: index number).  
- UNCT\_\[VAR\]: the uncertainties related to each variable where applicable, replacing \[VAR\] with the same name as above. If the uncertainty is not a distributed field but a global single value, it is in the attributes of the variable.


Each variable will come with, at least, a single scalar uncertainty value. The spatial resolution will vary from glacier to glacier (depending on the input data), between 2 and 50 m. Participants will not have access to the validation data.

## Double-anonymized experiment

The provided glaciers will not be named and the projections will be ‘anonymised’ by the WP1 team. The produced simulations will be anonymised before assessment by the WP2 & WP3 core team.

## Experiments

Experiments are shortly described here. An overview table is provided at the end of the document.

### WP2 \- Reference simulations 

Estimation of distributed surface mass balance based on homogenized input data (fully distributed input fields).  
Motivation: 

* test whether participant models perform well  
* real-world examples will show input robustness

### WP2 \- Simulations based on raw inputs

For the real-world cases, the input fields will be provided with existing gaps in the data, and only the GPR profiles of ice thickness will be provided. For the synthetic glacier setup, velocity and thickness values were reduced to lower elevations (below the 33%-percentile in the ice-covered area, 2914.6m a.s.l.) and to three elevation bands along the lower trunk (2300±5m, 2500±2.5m and 2700±5 m a.s.l.), respectively.

Motivation:

* Can some approaches directly deal with incomplete data?  
* Do some approaches have beneficial homogenisation capabilities in place?


### WP3 \- Simulations based on perturbed inputs

\~100 simulations for each of the glaciers with induced noise on various spatial scales

* 2 levels for ice thickness  
* 2 levels for velocity  
* reduced spatial resolution

Motivation:

* How sensible are the different methods to data degradation?  
* Can we infer a meaningful uncertainty bound from these perturbations?

### WP3 \- Globally-available inputs 

Only for the real-world cases. The input fields will be provided from globally-available datasets.  
Motivation:

* Can approaches deal with coarser & more uncertain data in view of a regional-to-global scale application? For instance by providing a reasonable mass balance gradient?

  # Receiving the data

* The data can be accessed at Zenodo: **10.5281/zenodo.21401808**   
* All the input data to represents \~10 GB (\~1-50 MB per glacier, per experiment).

# Validation

The outputs from the different models will be evaluated using statistical metrics (bias, R2, RMSE) against independent datasets (not necessarily distributed):

1. Stake SMB (distributed for the synthetic cases)  
2. Stake flux divergence (distributed for the synthetic cases)  
3. Glacier-wide geodetic MB

This evaluation will be conducted by the WP2+3 core team and we anticipate the preliminary results to be presented in a virtual meeting in January/February 2027, followed by an in-person meeting at EGU 2027, which will pave the way for the preparation of the ContinuIX paper to be submitted in 2027\.

# Who can participate in ContinuIX?

Anyone working with mass balance inversion methods and who is willing to run experiments on the provided data (and only using the provided data, i.e. without any additional meteorological or mass balance inputs), following requirements detailed below, is invited to participate. All contributors will be listed in the future publication as “method contributors”. 

Expectations of participants in WP2 and/or WP3:

1. General  
   1. Provide a detailed description of the applied methodology  
2. WP2  
   1. Provide results for all synthetic experiments  
   2. Provide results for the reference experiments for at least 4 real-world cases following a priority list (attached)  
   3. Provide results for all the ‘raw data’ experiments (optional)  
3. WP3   
   1. Provide results for all experiments for at least 4 real-world cases following a priority list (attached)  
   2. Provide results for the ‘globally-available data’  experiments for all the real-world cases (optional)

We further support all (data and method) contributors to become co-authors by fulfilling the additional two conditions:

1. You must read drafts and provide meaningful feedback. Potential co-authors will be given adequate time to provide this feedback.  
2. You must approve the submitted version prior to submission (and any substantially revised re-submitted versions).

# Submission deadline

**Intent to participate:** Please fill in the Google [Form](https://docs.google.com/forms/d/e/1FAIpQLSdOPA71TGtmtFK1WGLlxK5NUTMeYQBSR2UYkkisbPKoAyssRg/viewform?usp=publish-editor) by **August 31st.** We will then send you a link on which you will be able to upload your data and accompanying readmes.

**Data submission:** The simulations must be submitted in the prescribed format and received latest on **1 October 2026**. 

# Results submission

The following fields should be returned with their estimated uncertainty (if applicable), on the same georeferenced grid as the input data. This is best packaged as one netCDF file per glacier and per experiment, following the same conventions as the input data (including coordinate reference system), and the name, description and units given here:

* SMB: surface mass balance (unit: m i.e. yr-1)  
* FDIV: flux divergence (unit: m i.e. yr-1)  
* DHDT: elevation change if this was modified (unit: m i.e. yr-1)  
* VX: x-component of velocity if this was modified (unit: m yr-1)  
* VY: y-component of velocity if this was modified (unit: m yr-1)  
* THK: ice thickness if this was modified (unit: m i.e.)  
* DEM: surface topography if this was modified (unit: m a.s.l.)  
* BED: basal topography if this was modified (unit: m a.s.l.)  
* DENSITY: density (unit: kg m-3)  
* UNCT\_\[VAR\]: the uncertainties related to each variable where applicable, replacing \[VAR\] with the same name as above. If the uncertainty is not a distributed field but a global single value, put it in the attributes. We primarily require uncertainties on the SMB field, and if possible on the FDIV field

For flux-gate approaches, the surface mass balance values should be given for each of the provided bins.

You will be required to upload your results to the shared SharePoint folder (you will only be able to access your own folder) and to follow the following experiment submission structure:

   |-- GROUP\_abc/ 			\#\# recognizable group shorthand name   
       |--- EXP01/  
       	   |— EXP01\_G01\_method01.nc  	\#\# 1 netcdf per experiment, glacier and method  
       	   |— EXP01\_G02\_method01.nc  
       |--- EXP02/  
       	   |— EXP02\_G01\_method01.nc  
       	   |— EXP02\_G02\_method01.nc  
       |--- EXP03/  
       |--- log\_GROUP\_abc.txt              		 \#\# computation log, processing notes  
       |--- README\_GROUP\_abc\_method\#\#.txt   \#\# submitted metadata  
       |--- SUBMISSION\_CHECKLIST.txt       	 \#\# provided by ContinuIX team, to be filled out  
       |--- README\_submission\_template.txt 	\#\# provided by ContinuIX team: template  
       |--- FILE\_NAMING\_INSTRUCTIONS.txt   	\#\# provided by ContinuIX team

The results of each experiment for each glacier need to be submitted in one single netCDF file using the following naming convention: EXP\#\#\_G\#\#\_method\#\#.xxx, with EXP\#\# indicating the experiment number, G\#\# the glacier number and method\#\# the method number. Use 2 digit numbers. For example: EXP02\_G03\_method01.nc

Also required is the logged computation time and computing resources used to run the various experiments to be reported in a log\_GROUP\_abc.txt file.

In addition, please provide some additional information about the method(s) used as metadata in a  README\_GROUP\_hij\_method\#\#.txt file (one per method). This metadata should contain the following fields:

- METHOD DESCRIPTION: A concise paragraph (max. 10 sentences) detailing:  
  - Spatial discretization (e.g., flux gates, elevation bands, 2.5D grids).  
  - Flux divergence calculation scheme (e.g., analytical solution, optimization, ice flow model inversion).  
  - Surface mass balance calculation derivation.  
  - Density assumptions.  
  - Uncertainty estimation methodology.  
- DATA PRE-PROCESSING: Describe in detail all the pre-processing steps used to run the different experiments. List especially the name of the methods and resolution for any reprojection, resampling, gap filling or smoothing that you may have applied to any of the data fields before calculation of the flux divergence.  
- DOI / CITATION: If applicable, provide the DOI and full bibliographic reference of the original publication describing the method.  
- CONTRIBUTORS: Names and ORCID of the ‘method contributors’, i.e. who ran the method for the ContinuIX experiments and therefore quality to be co-authors on the ContinuIX paper(s)  
- INSTITUTION : Main institution(s)/affiliation responsible for the method development and application  
- ACKNOWLEDGEMENTS: List people and/or groups to acknowledge for the method development and application. Include contact info of corresponding author(s), including yourself (email).  
- PERMISSIONS: Indicate "confirm" that you agree that these outputs can be utilized in the ContinuIX project, and that analysis from this data may be disseminated in future ContinuIX publications. If preferred, you can specify a CC license.  
- ADDITIONAL INFORMATION: Any additional information you would like to provide regarding the method, results and their use.

# Contributors

The protocol was developed by (alphabetical order): Victor Devaux-Chupin, Johannes J. Fürst, Maaike Izeboud, Marin Kneib, Evan S. Miles, Albin Wells

# Contacts

Victor Devaux-Chupin: [vdevauxchupin@alaska.edu](mailto:vdevauxchupin@alaska.edu)   
Johannes J. Fürst: [johannes.fuerst@fau.de](mailto:johannes.fuerst@fau.de)   
Maaike Izeboud: [maaike.izeboud@vub.be](mailto:maaike.izeboud@vub.be)  
Marin Kneib: [marin.kneib@unifr.ch](mailto:marin.kneib@unifr.ch)   
Evan S. Miles: [miles@vaw.baug.ethz.ch](mailto:miles@vaw.baug.ethz.ch)   
Albin Wells: [albin.wells@geo.uzh.ch](mailto:albin.wells@geo.uzh.ch) 

# Acknowledgements

This document was inspired by the GlacierMIP4 and GlamBIE phase 2 protocols and by discussions with colleagues from the ITMIX and HistorIX working groups.

# Experiment overview table

| ID | Short name | Long name | Description | Mandatory glaciers | Which glaciers mandatory | Optional glaciers |
| :---- | :---- | :---- | :---- | :---: | ----- | ----: |
| Exp01 | REF | Reference experiment | Using homogenized data | 8 | real.+ synthetic | n/a |
| Exp02 | RAW | Raw experiments | Simulations based on raw inputs | 7 | real.+ synthetic | n/a |
| Exp03 | THK | Pertubation experiment for thickness | Exp3.1: correlation length 1 mean glacier thickness, random thickness error \+/-30% of local thickness (ITMIX1 result) | 4 | 2 real \+ 2 synthetic | 4 |
| Exp04 | THK | Pertubation experiment for thickness | Exp3.2: correlation length 2 mean glacier thickness, random thickness error \+/-30% of local thickness (ITMIX1 result) | 4 | 2 real \+ 2 synthetic | 4 |
| Exp05 | THK | Pertubation experiment for thickness | Exp3.3: correlation length 1 mean glacier thickness, random thickness error \+/-16% of mean thickness (ITMIX2 result) | 4 | 2 real \+ 2 synthetic | 4 |
| Exp06 | THK | Pertubation experiment for thickness | Exp3.4: absolute scaling of ice thickness \+30% | 4 | 2 real \+ 2 synthetic | 4 |
| Exp07 | THK | Pertubation experiment for thickness | Exp3.5: absolute scaling of ice thickness \-30% | 4 | 2 real \+ 2 synthetic | 4 |
| Exp08 | VEL | Pertubation experiment for velocity | Exp3.6: correlation length of 100m, random error of 10m/a (applied to magnitude \- no change in orientation | 4 | 2 real \+ 2 synthetic | 4 |
| Exp09 | VEL | Pertubation experiment for velocity | Exp3.7: correlation length of 1000m, random error of 10m/a (applied to magnitude \- no change in orientation | 4 | 2 real \+ 2 synthetic | 4 |
| Exp10 | VEL | Pertubation experiment for velocity | Exp3.8: correlation length of 100m, random error of 10% of pixel value (applied to magnitude \- no change in orientation | 4 | 2 real \+ 2 synthetic | 4 |
| Exp11 | VEL | Pertubation experiment for velocity | Exp3.9: correlation length of 100m, random error of 10% of pixel value applied separately to x and y components | 4 | 2 real \+ 2 synthetic | 4 |
| Exp12 | VEL | Pertubation experiment for velocity | Exp3.10: correlation length of 1000m, random error of 10% of pixel value applied separately to x and y components | 4 | 2 real \+ 2 synthetic | 4 |
| Exp13 | VEL | Pertubation experiment for velocity | Exp3.11: bias ramp scaling with altitude: 50% mean velocity of uncertainty at the highest altitude and 10% at the lowest altitude. | 4 | 2 real \+ 2 synthetic | 4 |
| Exp14 | RES | Pertubation experiment for resolution | Exp3.12: resolution reduction by factor of 3 in x and y (9 total) relative to original dataset | 4 | 2 real \+ 2 synthetic | 4 |
| Exp15 | RES | Pertubation experiment for resolution | Exp3.13: resolution to 100m posting (as for H2021 and many velocity products) | 4 | 2 real \+ 2 synthetic | 4 |
| Exp16 | GLOB | Experiment for alternative inputs, obtained from globally available data | Exp3.14: H2021 for dH, M2022 for u,v, but the site-specific thickness data | 6 | only real glaciers | n/a |
| Exp17 | GLOB | Experiment for alternative inputs, obtained from globally available data | Exp3.15: H2021 for dH, site-specific velocity data, Fa2019 for thickness | 6 | only real glaciers | n/a |
| Exp18 | GLOB | Experiment for alternative inputs, obtained from globally available data | Exp3.16: H2021 for dH, site-specific velocity data, Mi2022 for thickness | 6 | only real glaciers | n/a |
| Exp19 | GLOB | Experiment for alternative inputs, obtained from globally available data | Exp3.17: H2021 for dH, site-specific velocity data, Ma2025 for thickness | 6 | Only real glaciers | n/a |
| Exp20 | GLOB | Experiment for alternative inputs, obtained from globally available data | Exp3.18: H2021 for dH, M2022 velocity data, F2019 for thickness | 6 | only real glaciers | n/a |

