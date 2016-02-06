
include_directories(${CMAKE_SOURCE_DIR})
include("${CMAKE_SOURCE_DIR}/cmake/kernel.cmake")

# Makefile

if (DEFINED TARGET_CORE)
  #override CFLAGS += -DBUILD_KERNEL -DTABLE_NAME=gotoblas_$(TARGET_CORE)
  set(BUILD_KERNEL 1)
  set(KDIR "")
  set(TSUFFIX "_${TARGET_CORE}")
else ()
  set(TARGET_CORE ${CORE})
  set(KDIR "")
  set(TSUFFIX "")
endif ()

SetDefaultL1()
SetDefaultL2()
SetDefaultL3()
ParseMakefileVars("${KERNELDIR}/KERNEL")
ParseMakefileVars("${KERNELDIR}/KERNEL.${TARGET_CORE}")

if (${ARCH} STREQUAL "x86")
if (NOT MSVC)
  GenerateNamedObjects("${KERNELDIR}/cpuid.S" "" "" false "" "" true)
else()
  GenerateNamedObjects("${KERNELDIR}/cpuid_win.c" "" "" false "" "" true)
endif()
endif ()

# don't use float type name mangling here
GenerateNamedObjects("${KERNELDIR}/${LSAME_KERNEL}" "F_INTERFACE" "lsame" false "" "" true)
GenerateNamedObjects("${KERNELDIR}/${SCABS_KERNEL}" "COMPLEX;F_INTERFACE" "scabs1" false "" "" true)
GenerateNamedObjects("${KERNELDIR}/${DCABS_KERNEL}" "DOUBLE;COMPLEX;F_INTERFACE" "dcabs1" false "" "" true)

# Makefile.L1
foreach (float_type ${FLOAT_TYPES})
  # a bit of metaprogramming here to pull out the appropriate KERNEL var
  string(SUBSTRING ${float_type} 0 1 float_char)
  GenerateNamedObjects("${KERNELDIR}/${${float_char}AMAXKERNEL}" "USE_ABS" "amax_k" false "" "" false ${float_type})
  GenerateNamedObjects("${KERNELDIR}/${${float_char}AMINKERNEL}" "USE_ABS;USE_MIN" "amin_k" false "" "" false ${float_type})
  if (DEFINED ${float_char}MAXKERNEL)
    GenerateNamedObjects("${KERNELDIR}/${${float_char}MAXKERNEL}" "" "max_k" false "" "" false ${float_type})
  endif ()
  if (DEFINED ${float_char}MINKERNEL)
    GenerateNamedObjects("${KERNELDIR}/${${float_char}MINKERNEL}" "" "min_k" false "" "" false ${float_type})
  endif ()
  GenerateNamedObjects("${KERNELDIR}/${I${float_char}AMAXKERNEL}" "USE_ABS" "i*amax_k" false "" "" false ${float_type})
  GenerateNamedObjects("${KERNELDIR}/${I${float_char}AMINKERNEL}" "USE_ABS;USE_MIN" "i*amin_k" false "" "" false ${float_type})
  if (DEFINED I${float_char}MAXKERNEL)
    GenerateNamedObjects("${KERNELDIR}/${I${float_char}MAXKERNEL}" "" "i*max_k" false "" "" false ${float_type})
  endif ()
  if (DEFINED I${float_char}MINKERNEL)
    GenerateNamedObjects("${KERNELDIR}/${I${float_char}MINKERNEL}" "" "i*min_k" false "" "" false ${float_type})
  endif ()
  GenerateNamedObjects("${KERNELDIR}/${${float_char}ASUMKERNEL}" "" "asum_k" false "" "" false ${float_type})
  GenerateNamedObjects("${KERNELDIR}/${${float_char}AXPYKERNEL}" "" "axpy_k" false "" "" false ${float_type})
  GenerateNamedObjects("${KERNELDIR}/${${float_char}COPYKERNEL}" "C_INTERFACE" "copy_k" false "" "" false ${float_type})
  GenerateNamedObjects("${KERNELDIR}/${${float_char}NRM2KERNEL}" "" "nrm2_k" false "" "" false ${float_type})
  GenerateNamedObjects("${KERNELDIR}/${${float_char}ROTKERNEL}" "" "rot_k" false "" "" false ${float_type})
  GenerateNamedObjects("${KERNELDIR}/${${float_char}SCALKERNEL}" "" "scal_k" false "" "" false ${float_type})
  GenerateNamedObjects("${KERNELDIR}/${${float_char}SWAPKERNEL}" "" "swap_k" false "" "" false ${float_type})
  GenerateNamedObjects("${KERNELDIR}/${${float_char}AXPBYKERNEL}" "" "axpby_k" false "" "" false ${float_type})

  if (${float_type} STREQUAL "COMPLEX" OR ${float_type} STREQUAL "ZCOMPLEX")
    GenerateNamedObjects("${KERNELDIR}/${${float_char}AXPYKERNEL}" "CONJ" "axpyc_k" false "" "" false ${float_type})
    GenerateNamedObjects("${KERNELDIR}/${${float_char}DOTKERNEL}" "" "dotu_k" false "" "" false ${float_type})
    GenerateNamedObjects("${KERNELDIR}/${${float_char}DOTKERNEL}" "CONJ" "dotc_k" false "" "" false ${float_type})
  else ()
    GenerateNamedObjects("${KERNELDIR}/${${float_char}DOTKERNEL}" "" "dot_k" false "" "" false ${float_type})
  endif ()

  if (${float_type} STREQUAL "COMPLEX")
  GenerateNamedObjects("${KERNELDIR}/${${float_char}ROTKERNEL}" "" "srot_k" false "" "" false ${float_type})
  endif()
  if (${float_type} STREQUAL "ZCOMPLEX")
  GenerateNamedObjects("${KERNELDIR}/${${float_char}ROTKERNEL}" "" "drot_k" false "" "" false ${float_type})
  endif()

endforeach ()

#dsdot,sdsdot
GenerateNamedObjects("${KERNELDIR}/${DSDOTKERNEL}" "DSDOT" "d*dot_k" false "" "" false "SINGLE")
GenerateNamedObjects("${KERNELDIR}/${DSDOTKERNEL}" "DSDOT" "dsdot_k" false "" "" false "SINGLE")

# Makefile.L2
GenerateCombinationObjects("generic/symv_k.c" "LOWER" "U" "" 1 "" "" 3)
GenerateNamedObjects("generic/ger.c" "" "ger_k" false "" "" "" 3)
foreach (float_type ${FLOAT_TYPES})
  string(SUBSTRING ${float_type} 0 1 float_char)
  if (${float_type} STREQUAL "COMPLEX" OR ${float_type} STREQUAL "ZCOMPLEX")
    GenerateNamedObjects("${KERNELDIR}/${${float_char}GERUKERNEL}" "" "geru_k" false "" "" false ${float_type})
    GenerateNamedObjects("${KERNELDIR}/${${float_char}GERCKERNEL}" "CONJ" "gerc_k" false "" "" false ${float_type})
    GenerateNamedObjects("${KERNELDIR}/${${float_char}GERUKERNEL}" "XCONJ" "gerv_k" false "" "" false ${float_type})
    GenerateNamedObjects("${KERNELDIR}/${${float_char}GERCKERNEL}" "CONJ;XCONJ" "gerd_k" false "" "" false ${float_type})

    GenerateNamedObjects("${KERNELDIR}/${${float_char}GEMVNKERNEL}" "" "gemv_n" false "" "" false ${float_type})
    GenerateNamedObjects("${KERNELDIR}/${${float_char}GEMVTKERNEL}" "TRANSA" "gemv_t" false "" "" false ${float_type})
    GenerateNamedObjects("${KERNELDIR}/${${float_char}GEMVNKERNEL}" "CONJ" "gemv_r" false "" "" false ${float_type})
    GenerateNamedObjects("${KERNELDIR}/${${float_char}GEMVTKERNEL}" "CONJ;TRANSA" "gemv_c" false "" "" false ${float_type})
    GenerateNamedObjects("${KERNELDIR}/${${float_char}GEMVNKERNEL}" "XCONJ" "gemv_o" false "" "" false ${float_type})
    GenerateNamedObjects("${KERNELDIR}/${${float_char}GEMVTKERNEL}" "XCONJ;TRANSA" "gemv_u" false "" "" false ${float_type})
    GenerateNamedObjects("${KERNELDIR}/${${float_char}GEMVNKERNEL}" "XCONJ;CONJ" "gemv_s" false "" "" false ${float_type})
    GenerateNamedObjects("${KERNELDIR}/${${float_char}GEMVTKERNEL}" "XCONJ;CONJ;TRANSA" "gemv_d" false "" "" false ${float_type})

    GenerateNamedObjects("${KERNELDIR}/${${float_char}HEMV_U_KERNEL}" "HEMV" "hemv_U" false "" "" false ${float_type})
    GenerateNamedObjects("${KERNELDIR}/${${float_char}HEMV_L_KERNEL}" "HEMV;LOWER" "hemv_L" false "" "" false ${float_type})
    GenerateNamedObjects("${KERNELDIR}/${${float_char}HEMV_V_KERNEL}" "HEMV;HEMVREV" "hemv_V" false "" "" false ${float_type})
    GenerateNamedObjects("${KERNELDIR}/${${float_char}HEMV_M_KERNEL}" "HEMV;HEMVREV;LOWER" "hemv_M" false "" "" false ${float_type})

  else ()
    GenerateNamedObjects("${KERNELDIR}/${${float_char}GEMVNKERNEL}" "" "gemv_n" false "" "" false ${float_type})
    GenerateNamedObjects("${KERNELDIR}/${${float_char}GEMVTKERNEL}" "TRANS" "gemv_t" false "" "" false ${float_type})
  endif ()
endforeach ()

# Makefile.L3
set(USE_TRMM false)

if (${ARCH} STREQUAL "arm" OR ${ARCH} STREQUAL "arm64" OR "${TARGET}" STREQUAL "LONGSOON3B" OR "${TARGET}" STREQUAL "GENERIC" OR "${CORE}" STREQUAL "generic" OR "${TARGET}" STREQUAL "HASWELL" OR "${CORE}" STREQUAL "haswell")
  set(USE_TRMM true)
endif ()

foreach (float_type ${FLOAT_TYPES})
  string(SUBSTRING ${float_type} 0 1 float_char)
  GenerateNamedObjects("${KERNELDIR}/${${float_char}GEMMKERNEL}" "" "gemm_kernel" false "" "" false ${float_type})

  if (${float_char}GEMMINCOPY)
    GenerateNamedObjects("${KERNELDIR}/${${float_char}GEMMINCOPY}" "${float_type}" "${${float_char}GEMMINCOPYOBJ}" false "" "" true ${float_type})
  endif ()

  if (${float_char}GEMMITCOPY)
    GenerateNamedObjects("${KERNELDIR}/${${float_char}GEMMITCOPY}" "${float_type}" "${${float_char}GEMMITCOPYOBJ}" false "" "" true ${float_type})
  endif ()

  if (${float_char}GEMMONCOPY)
    GenerateNamedObjects("${KERNELDIR}/${${float_char}GEMMONCOPY}" "${float_type}" "${${float_char}GEMMONCOPYOBJ}" false "" "" true ${float_type})
  endif ()

  if (${float_char}GEMMOTCOPY)
    GenerateNamedObjects("${KERNELDIR}/${${float_char}GEMMOTCOPY}" "${float_type}" "${${float_char}GEMMOTCOPYOBJ}" false "" "" true ${float_type})
  endif ()

  GenerateNamedObjects("${KERNELDIR}/${${float_char}GEMM_BETA}" "" "gemm_beta" false "" "" false ${float_type})

  if (USE_TRMM)
    set(TRMM_KERNEL "${${float_char}TRMMKERNEL}")
  else ()
    set(TRMM_KERNEL "${${float_char}GEMMKERNEL}")
  endif ()

  if (${float_type} STREQUAL "COMPLEX" OR ${float_type} STREQUAL "ZCOMPLEX")

    # just enumerate all these. there is an extra define for these indicating which side is a conjugate (e.g. CN NC NN) that I don't really want to work into GenerateCombinationObjects

    GenerateNamedObjects("${KERNELDIR}/${${float_char}GEMMKERNEL}" "NN" "gemm_kernel_n" false "" "" false ${float_type})
    GenerateNamedObjects("${KERNELDIR}/${${float_char}GEMMKERNEL}" "CN" "gemm_kernel_l" false "" "" false ${float_type})
    GenerateNamedObjects("${KERNELDIR}/${${float_char}GEMMKERNEL}" "NC" "gemm_kernel_r" false "" "" false ${float_type})
    GenerateNamedObjects("${KERNELDIR}/${${float_char}GEMMKERNEL}" "CC" "gemm_kernel_b" false "" "" false ${float_type})

    GenerateNamedObjects("${KERNELDIR}/${TRMM_KERNEL}" "TRMMKERNEL;LEFT;NN" "trmm_kernel_LN" false "" "" false ${float_type})
    GenerateNamedObjects("${KERNELDIR}/${TRMM_KERNEL}" "TRMMKERNEL;LEFT;TRANSA;NN" "trmm_kernel_LT" false "" "" false ${float_type})
    GenerateNamedObjects("${KERNELDIR}/${TRMM_KERNEL}" "TRMMKERNEL;LEFT;CONJ;CN" "trmm_kernel_LR" false "" "" false ${float_type})
    GenerateNamedObjects("${KERNELDIR}/${TRMM_KERNEL}" "TRMMKERNEL;LEFT;TRANSA;CONJ;CN" "trmm_kernel_LC" false "" "" false ${float_type})
    GenerateNamedObjects("${KERNELDIR}/${TRMM_KERNEL}" "TRMMKERNEL;NN" "trmm_kernel_RN" false "" "" false ${float_type})
    GenerateNamedObjects("${KERNELDIR}/${TRMM_KERNEL}" "TRMMKERNEL;TRANSA;NN" "trmm_kernel_RT" false "" "" false ${float_type})
    GenerateNamedObjects("${KERNELDIR}/${TRMM_KERNEL}" "TRMMKERNEL;CONJ;NC" "trmm_kernel_RR" false "" "" false ${float_type})
    GenerateNamedObjects("${KERNELDIR}/${TRMM_KERNEL}" "TRMMKERNEL;TRANSA;CONJ;NC" "trmm_kernel_RC" false "" "" false ${float_type})

    GenerateNamedObjects("${KERNELDIR}/${${float_char}TRSMKERNEL_LN}" "UPPER;LN;TRSMKERNEL;CONJ" "trsm_kernel_LR" false "" "" false ${float_type})
    GenerateNamedObjects("${KERNELDIR}/${${float_char}TRSMKERNEL_LT}" "LT;TRSMKERNEL;CONJ" "trsm_kernel_LC" false "" "" false ${float_type})
    GenerateNamedObjects("${KERNELDIR}/${${float_char}TRSMKERNEL_RN}" "UPPER;RN;TRSMKERNEL;CONJ" "trsm_kernel_RR" false "" "" false ${float_type})
    GenerateNamedObjects("${KERNELDIR}/${${float_char}TRSMKERNEL_RT}" "RT;TRSMKERNEL;CONJ" "trsm_kernel_RC" false "" "" false ${float_type})


    #hemm
  GenerateNamedObjects("generic/zhemm_utcopy_${${float_char}GEMM_UNROLL_M}.c" "" "hemm_iutcopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/zhemm_ltcopy_${${float_char}GEMM_UNROLL_M}.c" "LOWER" "hemm_iltcopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/zhemm_utcopy_${${float_char}GEMM_UNROLL_N}.c" "OUTER" "hemm_outcopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/zhemm_ltcopy_${${float_char}GEMM_UNROLL_N}.c" "OUTER;LOWER" "hemm_oltcopy" false "" "" false ${float_type})

  # symm for c and z
  GenerateNamedObjects("generic/zsymm_ucopy_${${float_char}GEMM_UNROLL_N}.c" "OUTER" "symm_outcopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/zsymm_ucopy_${${float_char}GEMM_UNROLL_M}.c" "" "symm_iutcopy" false "" "" false ${float_type})

  GenerateNamedObjects("generic/zsymm_lcopy_${${float_char}GEMM_UNROLL_N}.c" "LOWER;OUTER" "symm_oltcopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/zsymm_lcopy_${${float_char}GEMM_UNROLL_M}.c" "LOWER" "symm_iltcopy" false "" "" false ${float_type})

  GenerateNamedObjects("generic/ztrmm_uncopy_${${float_char}GEMM_UNROLL_M}.c" "UNIT" "trmm_iunucopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/ztrmm_uncopy_${${float_char}GEMM_UNROLL_M}.c" "" "trmm_iunncopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/ztrmm_uncopy_${${float_char}GEMM_UNROLL_N}.c" "OUTER;UNIT" "trmm_ounucopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/ztrmm_uncopy_${${float_char}GEMM_UNROLL_N}.c" "OUTER" "trmm_ounncopy" false "" "" false ${float_type})

  GenerateNamedObjects("generic/ztrmm_lncopy_${${float_char}GEMM_UNROLL_M}.c" "LOWER;UNIT" "trmm_ilnucopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/ztrmm_lncopy_${${float_char}GEMM_UNROLL_M}.c" "LOWER" "trmm_ilnncopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/ztrmm_lncopy_${${float_char}GEMM_UNROLL_N}.c" "OUTER;LOWER;UNIT" "trmm_olnucopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/ztrmm_lncopy_${${float_char}GEMM_UNROLL_N}.c" "OUTER;LOWER" "trmm_olnncopy" false "" "" false ${float_type})

  GenerateNamedObjects("generic/ztrmm_utcopy_${${float_char}GEMM_UNROLL_M}.c" "UNIT" "trmm_iutucopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/ztrmm_utcopy_${${float_char}GEMM_UNROLL_M}.c" "" "trmm_iutncopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/ztrmm_utcopy_${${float_char}GEMM_UNROLL_N}.c" "OUTER;UNIT" "trmm_outucopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/ztrmm_utcopy_${${float_char}GEMM_UNROLL_N}.c" "OUTER" "trmm_outncopy" false "" "" false ${float_type})

  GenerateNamedObjects("generic/ztrmm_ltcopy_${${float_char}GEMM_UNROLL_M}.c" "LOWER;UNIT" "trmm_iltucopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/ztrmm_ltcopy_${${float_char}GEMM_UNROLL_M}.c" "LOWER" "trmm_iltncopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/ztrmm_ltcopy_${${float_char}GEMM_UNROLL_N}.c" "OUTER;LOWER;UNIT" "trmm_oltucopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/ztrmm_ltcopy_${${float_char}GEMM_UNROLL_N}.c" "OUTER;LOWER" "trmm_oltncopy" false "" "" false ${float_type})

  GenerateNamedObjects("generic/ztrsm_uncopy_${${float_char}GEMM_UNROLL_M}.c" "UNIT" "trsm_iunucopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/ztrsm_uncopy_${${float_char}GEMM_UNROLL_M}.c" "" "trsm_iunncopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/ztrsm_uncopy_${${float_char}GEMM_UNROLL_N}.c" "OUTER;UNIT" "trsm_ounucopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/ztrsm_uncopy_${${float_char}GEMM_UNROLL_N}.c" "OUTER" "trsm_ounncopy" false "" "" false ${float_type})

  GenerateNamedObjects("generic/ztrsm_lncopy_${${float_char}GEMM_UNROLL_M}.c" "LOWER;UNIT" "trsm_ilnucopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/ztrsm_lncopy_${${float_char}GEMM_UNROLL_M}.c" "LOWER" "trsm_ilnncopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/ztrsm_lncopy_${${float_char}GEMM_UNROLL_N}.c" "OUTER;LOWER;UNIT" "trsm_olnucopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/ztrsm_lncopy_${${float_char}GEMM_UNROLL_N}.c" "OUTER;LOWER" "trsm_olnncopy" false "" "" false ${float_type})

  GenerateNamedObjects("generic/ztrsm_utcopy_${${float_char}GEMM_UNROLL_M}.c" "UNIT" "trsm_iutucopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/ztrsm_utcopy_${${float_char}GEMM_UNROLL_M}.c" "" "trsm_iutncopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/ztrsm_utcopy_${${float_char}GEMM_UNROLL_N}.c" "OUTER;UNIT" "trsm_outucopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/ztrsm_utcopy_${${float_char}GEMM_UNROLL_N}.c" "OUTER" "trsm_outncopy" false "" "" false ${float_type})

  GenerateNamedObjects("generic/ztrsm_ltcopy_${${float_char}GEMM_UNROLL_M}.c" "LOWER;UNIT" "trsm_iltucopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/ztrsm_ltcopy_${${float_char}GEMM_UNROLL_M}.c" "LOWER" "trsm_iltncopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/ztrsm_ltcopy_${${float_char}GEMM_UNROLL_N}.c" "OUTER;LOWER;UNIT" "trsm_oltucopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/ztrsm_ltcopy_${${float_char}GEMM_UNROLL_N}.c" "OUTER;LOWER" "trsm_oltncopy" false "" "" false ${float_type})

  else ()  #For real
    GenerateCombinationObjects("${KERNELDIR}/${TRMM_KERNEL}" "LEFT;TRANSA" "R;N" "TRMMKERNEL" 2 "trmm_kernel" false ${float_type})

    # symm for s and d
    GenerateNamedObjects("generic/symm_ucopy_${${float_char}GEMM_UNROLL_N}.c" "OUTER" "symm_outcopy" false "" "" false ${float_type})
    GenerateNamedObjects("generic/symm_ucopy_${${float_char}GEMM_UNROLL_M}.c" "" "symm_iutcopy" false "" "" false ${float_type})

    GenerateNamedObjects("generic/symm_lcopy_${${float_char}GEMM_UNROLL_N}.c" "LOWER;OUTER" "symm_oltcopy" false "" "" false ${float_type})
    GenerateNamedObjects("generic/symm_lcopy_${${float_char}GEMM_UNROLL_M}.c" "LOWER" "symm_iltcopy" false "" "" false ${float_type})

  # These don't use a scheme that is easy to iterate over - the filenames have part of the DEFINE codes in them, for UPPER/TRANS but not for UNIT/OUTER. Also TRANS is not passed in as a define.
  # Could simplify it a bit by pairing up by -UUNIT/-DUNIT.

  GenerateNamedObjects("generic/trmm_uncopy_${${float_char}GEMM_UNROLL_M}.c" "UNIT" "trmm_iunucopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/trmm_uncopy_${${float_char}GEMM_UNROLL_M}.c" "" "trmm_iunncopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/trmm_uncopy_${${float_char}GEMM_UNROLL_N}.c" "OUTER;UNIT" "trmm_ounucopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/trmm_uncopy_${${float_char}GEMM_UNROLL_N}.c" "OUTER" "trmm_ounncopy" false "" "" false ${float_type})

  GenerateNamedObjects("generic/trmm_lncopy_${${float_char}GEMM_UNROLL_M}.c" "LOWER;UNIT" "trmm_ilnucopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/trmm_lncopy_${${float_char}GEMM_UNROLL_M}.c" "LOWER" "trmm_ilnncopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/trmm_lncopy_${${float_char}GEMM_UNROLL_N}.c" "OUTER;LOWER;UNIT" "trmm_olnucopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/trmm_lncopy_${${float_char}GEMM_UNROLL_N}.c" "OUTER;LOWER" "trmm_olnncopy" false "" "" false ${float_type})

  GenerateNamedObjects("generic/trmm_utcopy_${${float_char}GEMM_UNROLL_M}.c" "UNIT" "trmm_iutucopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/trmm_utcopy_${${float_char}GEMM_UNROLL_M}.c" "" "trmm_iutncopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/trmm_utcopy_${${float_char}GEMM_UNROLL_N}.c" "OUTER;UNIT" "trmm_outucopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/trmm_utcopy_${${float_char}GEMM_UNROLL_N}.c" "OUTER" "trmm_outncopy" false "" "" false ${float_type})

  GenerateNamedObjects("generic/trmm_ltcopy_${${float_char}GEMM_UNROLL_M}.c" "LOWER;UNIT" "trmm_iltucopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/trmm_ltcopy_${${float_char}GEMM_UNROLL_M}.c" "LOWER" "trmm_iltncopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/trmm_ltcopy_${${float_char}GEMM_UNROLL_N}.c" "OUTER;LOWER;UNIT" "trmm_oltucopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/trmm_ltcopy_${${float_char}GEMM_UNROLL_N}.c" "OUTER;LOWER" "trmm_oltncopy" false "" "" false ${float_type})

  GenerateNamedObjects("generic/trsm_uncopy_${${float_char}GEMM_UNROLL_M}.c" "UNIT" "trsm_iunucopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/trsm_uncopy_${${float_char}GEMM_UNROLL_M}.c" "" "trsm_iunncopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/trsm_uncopy_${${float_char}GEMM_UNROLL_N}.c" "OUTER;UNIT" "trsm_ounucopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/trsm_uncopy_${${float_char}GEMM_UNROLL_N}.c" "OUTER" "trsm_ounncopy" false "" "" false ${float_type})

  GenerateNamedObjects("generic/trsm_lncopy_${${float_char}GEMM_UNROLL_M}.c" "LOWER;UNIT" "trsm_ilnucopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/trsm_lncopy_${${float_char}GEMM_UNROLL_M}.c" "LOWER" "trsm_ilnncopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/trsm_lncopy_${${float_char}GEMM_UNROLL_N}.c" "OUTER;LOWER;UNIT" "trsm_olnucopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/trsm_lncopy_${${float_char}GEMM_UNROLL_N}.c" "OUTER;LOWER" "trsm_olnncopy" false "" "" false ${float_type})

  GenerateNamedObjects("generic/trsm_utcopy_${${float_char}GEMM_UNROLL_M}.c" "UNIT" "trsm_iutucopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/trsm_utcopy_${${float_char}GEMM_UNROLL_M}.c" "" "trsm_iutncopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/trsm_utcopy_${${float_char}GEMM_UNROLL_N}.c" "OUTER;UNIT" "trsm_outucopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/trsm_utcopy_${${float_char}GEMM_UNROLL_N}.c" "OUTER" "trsm_outncopy" false "" "" false ${float_type})

  GenerateNamedObjects("generic/trsm_ltcopy_${${float_char}GEMM_UNROLL_M}.c" "LOWER;UNIT" "trsm_iltucopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/trsm_ltcopy_${${float_char}GEMM_UNROLL_M}.c" "LOWER" "trsm_iltncopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/trsm_ltcopy_${${float_char}GEMM_UNROLL_N}.c" "OUTER;LOWER;UNIT" "trsm_oltucopy" false "" "" false ${float_type})
  GenerateNamedObjects("generic/trsm_ltcopy_${${float_char}GEMM_UNROLL_N}.c" "OUTER;LOWER" "trsm_oltncopy" false "" "" false ${float_type})

  endif ()

  GenerateNamedObjects("${KERNELDIR}/${${float_char}TRSMKERNEL_LN}" "UPPER;LN;TRSMKERNEL" "trsm_kernel_LN" false "" "" false ${float_type})
  GenerateNamedObjects("${KERNELDIR}/${${float_char}TRSMKERNEL_LT}" "LT;TRSMKERNEL" "trsm_kernel_LT" false "" "" false ${float_type})
  GenerateNamedObjects("${KERNELDIR}/${${float_char}TRSMKERNEL_RN}" "UPPER;RN;TRSMKERNEL" "trsm_kernel_RN" false "" "" false ${float_type})
  GenerateNamedObjects("${KERNELDIR}/${${float_char}TRSMKERNEL_RT}" "RT;TRSMKERNEL" "trsm_kernel_RT" false "" "" false ${float_type})



  if (NOT DEFINED ${float_char}OMATCOPY_CN)
    if (${float_char} STREQUAL "Z" OR ${float_char} STREQUAL "C")
      set(${float_char}OMATCOPY_CN ../arm/zomatcopy_cn.c)
    else ()
      set(${float_char}OMATCOPY_CN ../arm/omatcopy_cn.c)
    endif ()
  endif ()
  if (NOT DEFINED ${float_char}OMATCOPY_RN)
    if (${float_char} STREQUAL "Z" OR ${float_char} STREQUAL "C")
      set(${float_char}OMATCOPY_RN ../arm/zomatcopy_rn.c)
    else ()
      set(${float_char}OMATCOPY_RN ../arm/omatcopy_rn.c)
    endif  ()
  endif ()
  if (NOT DEFINED ${float_char}OMATCOPY_CT)
    if (${float_char} STREQUAL "Z" OR ${float_char} STREQUAL "C")
      set(${float_char}OMATCOPY_CT ../arm/zomatcopy_ct.c)
    else ()
      set(${float_char}OMATCOPY_CT ../arm/omatcopy_ct.c)
    endif ()
  endif ()
  if (NOT DEFINED ${float_char}OMATCOPY_RT)
    if (${float_char} STREQUAL "Z" OR ${float_char} STREQUAL "C")
      set(${float_char}OMATCOPY_RT ../arm/zomatcopy_rt.c)
    else ()
      set(${float_char}OMATCOPY_RT ../arm/omatcopy_rt.c)
    endif ()
  endif ()

  GenerateNamedObjects("${KERNELDIR}/${${float_char}OMATCOPY_CN}" "" "omatcopy_k_cn" false "" "" false ${float_type})
  GenerateNamedObjects("${KERNELDIR}/${${float_char}OMATCOPY_RN}" "ROWM" "omatcopy_k_rn" false "" "" false ${float_type})
  GenerateNamedObjects("${KERNELDIR}/${${float_char}OMATCOPY_CT}" "" "omatcopy_k_ct" false "" "" false ${float_type})
  GenerateNamedObjects("${KERNELDIR}/${${float_char}OMATCOPY_RT}" "ROWM" "omatcopy_k_rt" false "" "" false ${float_type})

  if (NOT DEFINED ${float_char}OMATCOPY_CNC)
    if (${float_char} STREQUAL "Z" OR ${float_char} STREQUAL "C")
      set(${float_char}OMATCOPY_CNC ../arm/zomatcopy_cnc.c)
    endif ()
  endif ()
  if (NOT DEFINED ${float_char}OMATCOPY_RNC)
    if (${float_char} STREQUAL "Z" OR ${float_char} STREQUAL "C")
      set(${float_char}OMATCOPY_RNC ../arm/zomatcopy_rnc.c)
    endif  ()
  endif ()
  if (NOT DEFINED ${float_char}OMATCOPY_CTC)
    if (${float_char} STREQUAL "Z" OR ${float_char} STREQUAL "C")
      set(${float_char}OMATCOPY_CTC ../arm/zomatcopy_ctc.c)
    endif ()
  endif ()
  if (NOT DEFINED ${float_char}OMATCOPY_RTC)
    if (${float_char} STREQUAL "Z" OR ${float_char} STREQUAL "C")
      set(${float_char}OMATCOPY_RTC ../arm/zomatcopy_rtc.c)
    endif ()
  endif ()

  if (${float_char} STREQUAL "Z" OR ${float_char} STREQUAL "C")
    GenerateNamedObjects("${KERNELDIR}/${${float_char}OMATCOPY_CNC}" "CONJ" "omatcopy_k_cnc" false "" "" false ${float_type})
    GenerateNamedObjects("${KERNELDIR}/${${float_char}OMATCOPY_RNC}" "CONJ;ROWM" "omatcopy_k_rnc" false "" "" false ${float_type})
    GenerateNamedObjects("${KERNELDIR}/${${float_char}OMATCOPY_CTC}" "CONJ" "omatcopy_k_ctc" false "" "" false ${float_type})
    GenerateNamedObjects("${KERNELDIR}/${${float_char}OMATCOPY_RTC}" "CONJ;ROWM" "omatcopy_k_rtc" false "" "" false ${float_type})
  endif()

  #imatcopy
  if (NOT DEFINED ${float_char}IMATCOPY_CN)
    if (${float_char} STREQUAL "Z" OR ${float_char} STREQUAL "C")
      set(${float_char}IMATCOPY_CN ../generic/zimatcopy_cn.c)
    else ()
      set(${float_char}IMATCOPY_CN ../generic/imatcopy_cn.c)
    endif ()
  endif ()

  if (NOT DEFINED ${float_char}IMATCOPY_RN)
    if (${float_char} STREQUAL "Z" OR ${float_char} STREQUAL "C")
      set(${float_char}IMATCOPY_RN ../generic/zimatcopy_rn.c)
    else ()
      set(${float_char}IMATCOPY_RN ../generic/imatcopy_rn.c)
    endif ()
  endif ()

  if (NOT DEFINED ${float_char}IMATCOPY_CT)
    if (${float_char} STREQUAL "Z" OR ${float_char} STREQUAL "C")
      set(${float_char}IMATCOPY_CT ../generic/zimatcopy_ct.c)
    else ()
      set(${float_char}IMATCOPY_CT ../generic/imatcopy_ct.c)
    endif ()
  endif ()

  if (NOT DEFINED ${float_char}IMATCOPY_RT)
    if (${float_char} STREQUAL "Z" OR ${float_char} STREQUAL "C")
      set(${float_char}IMATCOPY_RT ../generic/zimatcopy_rt.c)
    else ()
      set(${float_char}IMATCOPY_RT ../generic/imatcopy_rt.c)
    endif ()
  endif ()

  GenerateNamedObjects("${KERNELDIR}/${${float_char}IMATCOPY_CN}" "" "imatcopy_k_cn" false "" "" false ${float_type})
  GenerateNamedObjects("${KERNELDIR}/${${float_char}IMATCOPY_RN}" "ROWM" "imatcopy_k_rn" false "" "" false ${float_type})
  GenerateNamedObjects("${KERNELDIR}/${${float_char}IMATCOPY_CT}" "" "imatcopy_k_ct" false "" "" false ${float_type})
  GenerateNamedObjects("${KERNELDIR}/${${float_char}IMATCOPY_RT}" "ROWM" "imatcopy_k_rt" false "" "" false ${float_type})


  if (NOT DEFINED ${float_char}IMATCOPY_CNC)
    if (${float_char} STREQUAL "Z" OR ${float_char} STREQUAL "C")
      set(${float_char}IMATCOPY_CNC ../generic/zimatcopy_cnc.c)
    endif ()
  endif ()
  if (NOT DEFINED ${float_char}IMATCOPY_RNC)
    if (${float_char} STREQUAL "Z" OR ${float_char} STREQUAL "C")
      set(${float_char}IMATCOPY_RNC ../generic/zimatcopy_rnc.c)
    endif  ()
  endif ()
  if (NOT DEFINED ${float_char}IMATCOPY_CTC)
    if (${float_char} STREQUAL "Z" OR ${float_char} STREQUAL "C")
      set(${float_char}IMATCOPY_CTC ../generic/zimatcopy_ctc.c)
    endif ()
  endif ()
  if (NOT DEFINED ${float_char}IMATCOPY_RTC)
    if (${float_char} STREQUAL "Z" OR ${float_char} STREQUAL "C")
      set(${float_char}IMATCOPY_RTC ../generic/zimatcopy_rtc.c)
    endif ()
  endif ()

  if (${float_char} STREQUAL "Z" OR ${float_char} STREQUAL "C")
    GenerateNamedObjects("${KERNELDIR}/${${float_char}IMATCOPY_CNC}" "CONJ" "imatcopy_k_cnc" false "" "" false ${float_type})
    GenerateNamedObjects("${KERNELDIR}/${${float_char}IMATCOPY_RNC}" "CONJ;ROWM" "imatcopy_k_rnc" false "" "" false ${float_type})
    GenerateNamedObjects("${KERNELDIR}/${${float_char}IMATCOPY_CTC}" "CONJ" "imatcopy_k_ctc" false "" "" false ${float_type})
    GenerateNamedObjects("${KERNELDIR}/${${float_char}IMATCOPY_RTC}" "CONJ;ROWM" "imatcopy_k_rtc" false "" "" false ${float_type})
  endif()

  #geadd
  GenerateNamedObjects("${KERNELDIR}/${${float_char}GEADD_KERNEL}" "" "geadd_k" false "" "" false ${float_type})
endforeach ()

# Makefile.LA
#DBLASOBJS	+= dneg_tcopy$(TSUFFIX).$(SUFFIX) dlaswp_ncopy$(TSUFFIX).$(SUFFIX)

add_library(kernel OBJECT ${OPENBLAS_SRC})
