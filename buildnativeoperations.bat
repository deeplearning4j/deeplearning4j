#!/usr/bin/env bash

set OMP_NUM_THREADS=1

set num_args=0


if %num_args% LSS 1 (
    echo Please specify an argument
)
else (
    set command=%1
    echo Running  %1
    if  %1 == clean  (
       rd /s cmake_install.cmake
       rd /s cubinbuild
       rd /s ptxbuild
       rd /s CMakeFiles
       rm -f CMakeCache.txt
       rd /s testbuild
       rd /s eclipse/CMakeFiles

       echo Deleted build
       )
    else (
      if  %1 ==  eclipse  (
            cd eclipse
            cmake -GEclipse CDT4 - Unix Makefiles -DCMAKE_ECLIPSE_GENERATE_SOURCE_PROJECT=TRUE ..
            python ./nsight-err-parse-patch.py ./project
            move eclipse/.cproject .
            move eclipse/.project .
        )
    )
     else  (
     if %1 ==  lib (
        rd /s library  build
         mkdir librarybuild
         cd librarybuild
         cmake -DLIBRARY=TRUE ..
         make && cd ..
         )
     )

     else (
       if %1 ==  test (
           if  num_args GTR 1  (
                rd /s testbuild
                mkdir testbuild
                cd testbuild
                cmake -DRUN_TEST=TRUE ..
                make && cd ..
                move testbuild/test/libnd4jtests .
               ./libnd4jtests -n %2
               )
           else (
               rd /s testbuild
               mkdir testbuild
               cd testbuild
               cmake -DRUN_TEST=TRUE ..
               make && cd ..
               move testbuild/test/libnd4jtests .
               ./libnd4jtests
           )

           echo FINISHING BUILD
           )
      )
     else (
       if %1 == cubin  (
            rd /s cubinbuild
           mkdir cubinbuild
           cd cubinbuild
           cmake -DCUBIN=TRUE ..
           make && cd ..
           echo FINISHING BUILD
           move cubinbuild/cubin/cuda_compile_cubin_generated_all.cu.cubin all.cubin
    )
 )

      else (
      if %1 == buffer (
            rd /s bufferbuild
           mkdir bufferbuild
           cd bufferbuild
           cmake -DBUFFER=TRUE ..
           make && cd ..
           echo FINISHING BUILD
        )
        )
     else (
     if %1 == blas (
            rd /s blasbuild
           mkdir blasbuild
           cd blasbuild
           if  %num_args% GTR 1 (
              if %2 == cuda  (
                   cmake -DCUDA_BLAS=true -DBLAS=TRUE ..
                   make && cd ..
                  echo FINISHING BUILD
                  )
              else (
                 if %2 == cpu  (
                   cmake -DCPU_BLAS=true -DBLAS=TRUE ..
                   make && cd ..
                   echo FINISHING BUILD
                )
               )
              else (
                   echo Please specify cpu or gpu
                )
              )

            else (
                cmake  -DCPU_BLAS=true -DBLAS=TRUE ..
                  make && cd ..
                  echo FINISHING BUILD
            )

   )

      else (
      if %1 == ptx  (
           rd /s ptxbuild
           mkdir ptxbuild
           cd ptxbuild
           cmake -DPTX=TRUE ..
           make && cd ..
           echo FINISHING BUILD
           move ptxbuild/ptx/cuda_compile_ptx_generated_all.cu.ptx all.ptx
       )
    )
     else (
      if  %1 == fatbin (
           rd /s fatbuild
           mkdir fatbuild
           cd fatbuild
           cmake -DFATBIN=TRUE ..
           make && cd ..
           echo FINISHING BUILD
           move fatbuild/fatbin/cuda_compile_fatbin_generated_all.cu.fatbin all.fatbin
)
)

)
