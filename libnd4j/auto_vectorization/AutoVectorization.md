# Auto-vectorization Report

This report tool is used to get a human-friendly compiler output of the auto-vectorization process. It is intended for developers to help them to investigate the obstacles that compiler faced during auto-vectorization.

## Usage
```--check-vectorization``` option should be added to the **release** build to be able to get the auto-vectorization report 
```./buildnativeoperations.sh -a native  -j 28 --check-vectorization```
it will output ```vecmiss.html``` inside  blasbuild/cpu folder.

## Report Format 
Each filename contains info about optimization attempts for the source code lines. 
Each line number is also expandable (⇲) and contains distinct failure notes. 
It is possible to click on the line number to see source code  

| file name  | total successful attempts | total failed attempts | ⇲  |
|---|---|---|--|
| line number  | successful attempts  | failed attempts  | ⇲ |
|- failure reasons  |
| line number  | successful attempts  | failed attempts  |⇲ |

##### Requirements 
- GCC (Currently, only GCC is supported)
- python3

### Detailed report with `-fsave-optimization-record` option:
If you want to get more detailed information (for now it reports the functions of failures)  you should use new version of the toolchain (GCC > 9). As the new version of GCC compilers have `-fsave-optimization-record` option.   
`buildnativeoperations.sh` using CMake will detect it and switch to the more detailed version.
Please, note that this option is still experimental and so the compiler can fail to output some json.gz file with error.
On that case try to exclude those files from the build.  
And also the internal structure of the `-fsave-optimization-record` json.gz  can be changed in future. 

It outputs two files **vecmiss_fsave.html** and **vecmiss_fsave.html.js**. So to see report details you need to enable javascript on browser if it was disabled.

##### Requirements for the Detailed report
- GCC version > 9
- python3 
- Cython (python3)
- json   (python3)
- gzip   (python3) 
- c++filt

Internally, we are using Cython to speed up json.gz file processing (bigGzipJson.pyx). Because json.gz files can take big memory in raw when loaded in whole.

If you want to use bigGzipJson outside  `buildnativeoperations.sh` and CMake then you should compile it manually using this command in auto_vectorization folder:
`python3 cython_setup.py build_ext --inplace`  

json.gz files could be processed outside of `buildnativeoperations.sh`.    
You need to call `python3 auto_vect.py --fsave` inside base source folder and where json.gz files exist.  

