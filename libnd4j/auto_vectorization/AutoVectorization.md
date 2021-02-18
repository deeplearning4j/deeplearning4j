
# Auto-vectorization Report

This report tool is used to get a human-friendly compiler output of the auto-vectorization process. It is intended for developers to help them to investigate the obstacles that compiler faced during auto-vectorization.

## Usage
```--check-vectorization``` option should be added to the **release** build to be able to get the auto-vectorization report 
```./buildnativeoperations.sh -a native  -j 28 --check-vectorization```
it will output ```vecmiss.html``` inside  blasbuild/cpu folder.    
 
For the direct usage:
`compile command | python3 auto_vect.py` 
Also please note that to use it with `parallel make` one should add  `--output-sync=target`

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

##### Adding new  compiler support for the stdin message parsing
To add new compiler for the stdin processing one should add entry in `STDIN_COMPILER_ENTRY` for that compiler with the following syntax

       { 'compiler_name' : [('comparision', 'version with dot delimiter', 'entry_name') ,  other version and etc] }  
       example: STDIN_COMPILER_ENTRY = { 'gcc' : [('<','9','gcc_old'),...] ,...}

   The next step to add a parser for the entry in `STDIN_PARSERS`  
 `  STDIN_PARSERS = { 'gcc_old' : parser_method }`
 the signature of the parser function is:  
 `Parse_info parser_method(line, helper_storage)`  
- the line is a compiler output that needs to be parsed. 
- helper_storage is a dict and can be used as a state storage to parse multi-line and et cetera, as parser called for each line.
- Please note that Parse_info members should be the same with those which were defined in `general_stdin_parser local_parser`  

to  simplify adding compiler, especially, for those which outputs message details in one line, there is the helper method `general_stdin_parser("succes hint in the message", "failure hint in the message",  (file, line, message) extractor regex pattern)`:

    example: general_stdin_parser("vectorized loop", "unvectorized loop", r'[^/]+([^,]+)\,\s*line\s*(\d+)\:(.*)')


### Detailed report with `-fsave-optimization-record` option:
If you want to get more detailed information (for now it reports the functions of failures)  you should use new version of the toolchain (GCC > 9). As the new version of GCC compilers have `-fsave-optimization-record` option.   
`buildnativeoperations.sh` using CMake will detect it and switch to the more detailed version.
Please, note that this option is still experimental and so the compiler can fail to output some json.gz file with error.
On that case try to exclude those files from the build.  
And also the internal structure of the `-fsave-optimization-record` json.gz  can be changed in future. 

It outputs two files **vecmiss_fsave.html** and **vecmiss_fsave.html.js**. So to see report details you need to enable javascript on browser if it was disabled.

There is also `--inverted-file` option to generate inverted index for optimization messages in json format **vecmiss_fsave_inverted_index.json**. 
`inverted_index.py` script contains methods to work with those generated json outputs. For now one can get postings for optimization messages and filter those message based on file index and function index. File and function index can be obtained using the methods with  a predicate filter .

    message : [ file_index, line_position, [ compressed list of function index] ]
#### Requirements for the Detailed report
- GCC version > 9
- python3 
- Cython (python3)
- json   (python3)
- gzip   (python3) 
- c++filt

##### Some internal notes for `-fsave-optimization-record`  output format handling
Internally, we are using Cython to speed up json.gz file processing (bigGzipJson.pyx). Because json.gz files can take big memory in raw when loaded in whole.

If you want to use bigGzipJson outside  `buildnativeoperations.sh` and CMake then you should compile it manually using this command in auto_vectorization folder:
`python3 cython_setup.py build_ext --inplace`  

json.gz files could be processed outside of `buildnativeoperations.sh`.    
You need to call `python3 auto_vect.py --fsave` inside base source folder and where json.gz files exist. 
