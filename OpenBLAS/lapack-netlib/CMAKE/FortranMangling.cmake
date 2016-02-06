# Macro that defines variables describing the Fortran name mangling
# convention
#
# Sets the following outputs on success:
#
#  INTFACE
#    Add_
#    NoChange
#    f77IsF2C
#    UpCase
#    
macro(FORTRAN_MANGLING CDEFS)
MESSAGE(STATUS "=========")
  GET_FILENAME_COMPONENT(F77_NAME ${CMAKE_Fortran_COMPILER} NAME)
  GET_FILENAME_COMPONENT(F77_PATH ${CMAKE_Fortran_COMPILER} PATH)
  SET(F77 ${F77_NAME} CACHE INTERNAL "Name of the fortran compiler.")

  IF(${F77} STREQUAL "ifort.exe")
    #settings for Intel Fortran
    SET(F77_OPTION_COMPILE "/c" CACHE INTERNAL
      "Fortran compiler option for compiling without linking.")
    SET(F77_OUTPUT_OBJ "/Fo" CACHE INTERNAL
      "Fortran compiler option for setting object file name.")
    SET(F77_OUTPUT_EXE "/Fe" CACHE INTERNAL
      "Fortran compiler option for setting executable file name.")
  ELSE(${F77} STREQUAL "ifort.exe")
    # in other case, let user specify their fortran configrations.
    SET(F77_OPTION_COMPILE "-c" CACHE STRING
      "Fortran compiler option for compiling without linking.")
    SET(F77_OUTPUT_OBJ "-o" CACHE STRING
      "Fortran compiler option for setting object file name.")
    SET(F77_OUTPUT_EXE "-o" CACHE STRING
      "Fortran compiler option for setting executable file name.")
    SET(F77_LIB_PATH "" CACHE PATH
      "Library path for the fortran compiler")
    SET(F77_INCLUDE_PATH "" CACHE PATH
      "Include path for the fortran compiler")
  ENDIF(${F77} STREQUAL "ifort.exe")


MESSAGE(STATUS "Testing FORTRAN_MANGLING")
       
MESSAGE(STATUS "Compiling Finface.f...")

    execute_process ( COMMAND  ${CMAKE_Fortran_COMPILER} ${F77_OPTION_COMPILE} ${PROJECT_SOURCE_DIR}/lapacke/mangling/Fintface.f
      WORKING_DIRECTORY  ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp
      OUTPUT_VARIABLE    OUTPUT
      RESULT_VARIABLE    RESULT
      ERROR_VARIABLE     ERROR)

    if(RESULT EQUAL 0)
    MESSAGE(STATUS "Compiling Finface.f successful")
    else()
    MESSAGE(FATAL_ERROR " Compiling Finface.f FAILED")
    MESSAGE(FATAL_ERROR " Error:\n ${ERROR}")
    endif()

MESSAGE(STATUS "Compiling Cintface.c...")

    execute_process ( COMMAND  ${CMAKE_C_COMPILER} ${F77_OPTION_COMPILE} ${PROJECT_SOURCE_DIR}/lapacke/mangling/Cintface.c
      WORKING_DIRECTORY  ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp
      OUTPUT_VARIABLE    OUTPUT
      RESULT_VARIABLE    RESULT
      ERROR_VARIABLE     ERROR)

    if(RESULT EQUAL 0)
    MESSAGE(STATUS "Compiling Cintface.c successful")
    else()
    MESSAGE(FATAL_ERROR " Compiling Cintface.c FAILED")
    MESSAGE(FATAL_ERROR " Error:\n ${ERROR}")
    endif()

MESSAGE(STATUS "Linking Finface.f and Cintface.c...")

    execute_process ( COMMAND  ${CMAKE_Fortran_COMPILER} ${F77_OUTPUT_OBJ} xintface.exe Fintface.o Cintface.o
      WORKING_DIRECTORY  ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp
      OUTPUT_VARIABLE    OUTPUT
      RESULT_VARIABLE    RESULT
      ERROR_VARIABLE     ERROR)

    if(RESULT EQUAL 0)
    MESSAGE(STATUS "Linking Finface.f and Cintface.c successful")
    else()
    MESSAGE(FATAL_ERROR " Linking Finface.f and Cintface.c FAILED")
    MESSAGE(FATAL_ERROR " Error:\n ${ERROR}")
    endif()

MESSAGE(STATUS "Running ./xintface...")

    execute_process ( COMMAND  ./xintface.exe
      WORKING_DIRECTORY  ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp
      RESULT_VARIABLE xintface_RES
      OUTPUT_VARIABLE xintface_OUT
      ERROR_VARIABLE xintface_ERR)
                         

       if (xintface_RES EQUAL 0)
          STRING(REPLACE "\n" "" xintface_OUT "${xintface_OUT}")
          MESSAGE(STATUS "Fortran MANGLING convention: ${xintface_OUT}")
          SET(CDEFS ${xintface_OUT})
      else()
          MESSAGE(FATAL_ERROR "FORTRAN_MANGLING:ERROR ${xintface_ERR}")
      endif() 
      
endmacro(FORTRAN_MANGLING)
