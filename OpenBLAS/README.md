# OpenBLAS

[![Join the chat at https://gitter.im/xianyi/OpenBLAS](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/xianyi/OpenBLAS?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Travis CI: [![Build Status](https://travis-ci.org/xianyi/OpenBLAS.png?branch=develop)](https://travis-ci.org/xianyi/OpenBLAS)

AppVeyor: [![Build status](https://ci.appveyor.com/api/projects/status/09sohd35n8nkkx64/branch/develop?svg=true)](https://ci.appveyor.com/project/xianyi/openblas/branch/develop)
## Introduction
OpenBLAS is an optimized BLAS library based on GotoBLAS2 1.13 BSD version.

Please read the documents on OpenBLAS wiki pages <http://github.com/xianyi/OpenBLAS/wiki>.

## Binary Packages
We provide binary packages for the following platform.

  * Windows x86/x86_64

You can download them from [file hosting on sourceforge.net](https://sourceforge.net/projects/openblas/files/).

## Installation from Source
Download from project homepage. http://xianyi.github.com/OpenBLAS/

Or, check out codes from git://github.com/xianyi/OpenBLAS.git
### Normal compile
  * type "make" to detect the CPU automatically.
  or
  * type "make TARGET=xxx" to set target CPU, e.g. "make TARGET=NEHALEM". The full target list is in file TargetList.txt.

### Cross compile
Please set CC and FC with the cross toolchains. Then, set HOSTCC with your host C compiler. At last, set TARGET explicitly.

Examples:

On X86 box, compile this library for loongson3a CPU.

    make BINARY=64 CC=mips64el-unknown-linux-gnu-gcc FC=mips64el-unknown-linux-gnu-gfortran HOSTCC=gcc TARGET=LOONGSON3A

On X86 box, compile this library for loongson3a CPU with loongcc (based on Open64) compiler.

    make CC=loongcc FC=loongf95 HOSTCC=gcc TARGET=LOONGSON3A CROSS=1 CROSS_SUFFIX=mips64el-st-linux-gnu-   NO_LAPACKE=1 NO_SHARED=1 BINARY=32

### Debug version

    make DEBUG=1

### Install to the directory (optional)

Example:

    make install PREFIX=your_installation_directory

The default directory is /opt/OpenBLAS

## Support CPU & OS
Please read GotoBLAS_01Readme.txt

### Additional support CPU:

#### x86/x86-64:
- **Intel Xeon 56xx (Westmere)**: Used GotoBLAS2 Nehalem codes.
- **Intel Sandy Bridge**: Optimized Level-3 and Level-2 BLAS with AVX on x86-64.
- **Intel Haswell**: Optimized Level-3 and Level-2 BLAS with AVX2 and FMA  on x86-64.
- **AMD Bobcat**: Used GotoBLAS2 Barcelona codes.
- **AMD Bulldozer**: x86-64 ?GEMM FMA4 kernels. (Thank Werner Saar)
- **AMD PILEDRIVER**: Uses Bulldozer codes with some optimizations.
- **AMD STEAMROLLER**: Uses Bulldozer codes with some optimizations.

#### MIPS64:
- **ICT Loongson 3A**: Optimized Level-3 BLAS and the part of Level-1,2.
- **ICT Loongson 3B**: Experimental

#### ARM:
- **ARMV6**: Optimized BLAS for vfpv2 and vfpv3-d16 ( e.g. BCM2835, Cortex M0+ )
- **ARMV7**: Optimized BLAS for vfpv3-d32 ( e.g. Cortex A8, A9 and A15 )

#### ARM64:
- **ARMV8**: Experimental

### Support OS:
- **GNU/Linux**
- **MingWin/Windows**: Please read <https://github.com/xianyi/OpenBLAS/wiki/How-to-use-OpenBLAS-in-Microsoft-Visual-Studio>.
- **Darwin/Mac OS X**: Experimental. Although GotoBLAS2 supports Darwin, we are the beginner on Mac OS X.
- **FreeBSD**: Supported by community. We didn't test the library on this OS.

## Usages
Link with libopenblas.a or -lopenblas for shared library.

### Set the number of threads with environment variables.

Examples:

    export OPENBLAS_NUM_THREADS=4

 or

    export GOTO_NUM_THREADS=4

 or

    export OMP_NUM_THREADS=4

The priorities are OPENBLAS_NUM_THREADS > GOTO_NUM_THREADS > OMP_NUM_THREADS.

If you compile this lib with USE_OPENMP=1, you should set OMP_NUM_THREADS environment variable. OpenBLAS ignores OPENBLAS_NUM_THREADS and GOTO_NUM_THREADS with USE_OPENMP=1.

### Set the number of threads on runtime.

We provided the below functions to control the number of threads on runtime.

    void goto_set_num_threads(int num_threads);

    void openblas_set_num_threads(int num_threads);

If you compile this lib with USE_OPENMP=1, you should use the above functions, too.

## Report Bugs
Please add a issue in https://github.com/xianyi/OpenBLAS/issues

## Contact
* OpenBLAS users mailing list: https://groups.google.com/forum/#!forum/openblas-users
* OpenBLAS developers mailing list: https://groups.google.com/forum/#!forum/openblas-dev

## ChangeLog
Please see Changelog.txt to obtain the differences between GotoBLAS2 1.13 BSD version.

## Troubleshooting
* Please read [Faq](https://github.com/xianyi/OpenBLAS/wiki/Faq) at first.
* Please use gcc version 4.6 and above to compile Sandy Bridge AVX kernels on Linux/MingW/BSD.
* Please use Clang version 3.1 and above to compile the library on Sandy Bridge microarchitecture. The Clang 3.0 will generate the wrong AVX binary code.
* The number of CPUs/Cores should less than or equal to 256. On Linux x86_64(amd64), there is experimental support for up to 1024 CPUs/Cores and 128 numa nodes if you build the library with BIGNUMA=1.
* OpenBLAS does not set processor affinity by default. On Linux, you can enable processor affinity by commenting the line NO_AFFINITY=1 in Makefile.rule. But this may cause [the conflict with R parallel](https://stat.ethz.ch/pipermail/r-sig-hpc/2012-April/001348.html).
* On Loongson 3A. make test would be failed because of pthread_create error. The error code is EAGAIN. However, it will be OK when you run the same testcase on shell.

## Contributing
1. [Check for open issues](https://github.com/xianyi/OpenBLAS/issues) or open a fresh issue to start a discussion around a feature idea or a bug.
1. Fork the [OpenBLAS](https://github.com/xianyi/OpenBLAS) repository to start making your changes.
1. Write a test which shows that the bug was fixed or that the feature works as expected.
1. Send a pull request. Make sure to add yourself to `CONTRIBUTORS.md`.

## Donation
Please read [this wiki page](https://github.com/xianyi/OpenBLAS/wiki/Donation).
