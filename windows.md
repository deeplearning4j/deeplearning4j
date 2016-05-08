# Building on Windows
*All of these instructions assume you are on a 64-bit system*

libnd4j depends on some Unix utilities for compilation. So in order to compile it you will need to install  [Msys2](https://msys2.github.io/).

After you have setup Msys2 by following [their instructions](https://msys2.github.io/), you will have to install some additional development packages. Start the msys2 shell and setup the dev environment with:

    pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-extra-cmake-modules make pkg-config grep sed gzip tar mingw64/mingw-w64-x86_64-openblas

This will install the needed dependencies for use in the msys2 shell.

You will also need to setup your PATH environment variable to include `C:\msys64\mingw64\bin` (or where ever you have decided to install msys2). If you have IntelliJ (or another IDE) open, you will have to restart it before this change takes effect for applications started through them. If you don't, you probably will see a "Can't find dependent libraries" error.

For cpu, we recommend openblas. We will be adding instructions for mkl and other cpu implementations later.

Send us a pull request or [file an issue](https://github.com/deeplearning4j/libnd4j/issues) if you have something in particular you are looking for.

## Building libnd4j

libnd4j and nd4j go hand in hand, and libnd4j is required for two out of the three currently supported backends (nd4j-native and nd4j-cuda). For this reason they should always be rebuild together.

### Building the CPU Backend

Now clone this repository, and in that directory run the following to build the dll for the cpu backend:

    ./buildnativeoperations.sh

### Building the CUDA Backend

The CUDA Backend has some additional requirements before it can be built:

* [CUDA SDK](https://developer.nvidia.com/cuda-downloads)
* [Visual Studio 2012 or 2013](https://www.visualstudio.com/en-us/news/vs2013-community-vs.aspx) (Please note: Visual Studio 2015 is *NOT SUPPORTED* by CUDA 7.5 and below)

In order to build the CUDA backend you will have to setup some more environment variables first, by calling `vcvars64.bat`.
But first, set the system environment variable `SET_FULL_PATH` to `true`, so all of the variables that `vcvars64.bat` sets up, are passed to the mingw shell.

1. Inside a normal cmd.exe command prompt, run `C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\amd64\vcvars64.bat`
2. Run `c:\msys64\mingw64_shell.bat` inside that
3. Change to your libnd4j folder
4. `./buildnativeoperations.sh -c cuda`

This builds the CUDA nd4j.dll.


## Building nd4j

While still in the `libnd4j` folder, run:

    export LIBND4J_HOME=`pwd`

    or

    If you want to use Control Panel for that: if you have libnd4j path looking like 'c:\Users\username\libnd4j' set LIBND4J_HOME to '/Users/username/libnd4j'

Now leave the libnd4j directory and clone the [nd4j repository](https://github.com/deeplearning4j/nd4j). Run the following to compile nd4j with support for both the native cpu backend as well as the cuda backend:

    mvn clean install -DskipTests -Dmaven.javadoc.skip=true

If you don't want the cuda backend, e.g. because you didn't or can't build it, you can skip it:

    mvn clean install -DskipTests -Dmaven.javadoc.skip=true -pl '!org.nd4j:nd4j-cuda-7.5'

Please notice the single quotes around the last parameter, if you leave them out or use double quotes you will get an error about `event not found` from your shell. If this doesn't work, make sure you have a current version of maven installed.


## Using the Native Backend

In order to use your new shiny backends you will have to switch your application to use the version of ND4J that you just compiled and to use the native backend instead of x86.

For this you change the version of all your ND4J dependencies to "0.4-rc3.9-SNAPSHOT".


### CPU Backend

Exchange nd4j-x86 for nd4j-native like that:

    <dependency>
        <groupId>org.nd4j</groupId>
        <artifactId>nd4j-native</artifactId>
        <version>0.4-rc3.9-SNAPSHOT</version>
    </dependency>

### CUDA Backend

Exchange nd4j-x86 for nd4j-cuda-7.5 like that:

    <dependency>
        <groupId>org.nd4j</groupId>
        <artifactId>nd4j-cuda-7.5</artifactId>
        <version>0.4-rc3.9-SNAPSHOT</version>
    </dependency>

    
## Troubleshooting

### When I start my application, I still see a "Can't find dependent libraries" error
If your application continues to run, then you are just seeing an artefact of one way we try to load the native library, but your application should run just fine.

If your application crashes (and you see that error more than once) then you probably have a problem with your PATH environment variable. Please make sure that you have your msys2 bin directory on the PATH and that you restarted your IDE. Maybe even try to restart the system.

### I'm having trouble downloading or updating packages using pacman
There are a number of things that can potentially go wrong.
First, try updating packman using the following commands:

	pacman -Syy
	pacman -Syu
	pacman -S pacman-mirrors

Note that you might need to restart the msys2 shell between/after these steps.

One user has reported issues downloading packages using the default downloader (timeouts and "error: failed retrieving file" messages). If you are experiencing these issues, it may help to switch to using the wget downloader. To do this, install wget using

	pacman -S wget

then uncomment (remove the # symbol) the following line in the /etc/pacman.conf configuration file:

	XferCommand = /usr/bin/wget --passive-ftp -c -O %o %u

### "buildnativeoperations.sh blas cpu" can't find BLAS libraries

First, make sure you have BLAS libraries intalled. Typically, this involves building OpenBLAS by downloading OpenBLAS and running the commands 'make', 'make install' in msys2.

Running the buildnativeoperations.sh script in the MinGW-w64 Win64 Shell instead of the standard msys2 shell may resolve this issue.

### I'm getting other errors not listed here

Depending on how your build environment and PATH environment variable is set up, you might experience some other issues.
Some situations that may be problematic include:

- Having older (or multiple) MinGW installs on your PATH (check: type "where c++" or "where gcc" into msys2)
- Having older (or multiple) cmake installs on your PATH (check: "where cmake" and "cmake --version")
- Having multiple BLAS libraries on your PATH (check: "where libopenblas.dll", "where libblas.dll" and "where liblapack.dll")

### I'm getting `jniNativeOps.dll: Can't find dependent libraries` errors

This is usually due to an incorrectly setup PATH (see "I'm getting other errors not listed here"). As the PATH using the msys2 shell is a little bit different then for other applications, you can check that the PATH is really the problem by running the following test program:

    public class App {
        public static void main(String[] args){
        	System.loadLibrary("libopenblas.dll");
        }
    }
    
If this also crashes with the `Can't find dependent libraries` error, then you have to setup your PATH correctly (see the introduction to this document).


**Note**: Another possible cause of "...jniNativeOps.dll: Can't find dependent libraries" seems to be having an old or incompatible version of libstc++-6.dll on your PATH. You want this file to be pulled in from mingw via you PATH environment variable.
To check your PATH/environment, run `where libstdc++-6.dll` and `where libgcc_s_seh-1.dll`; these should list the msys/mingw directories (and/or **list them first**, if there are other copies on the PATH).


Finally, using dumpbin (from Visual Studio) can help to show required dependencies for jniNativeOps.dll:

	dumpbin /dependents [path to jniNativeOps.dll]
	
### My application crashes on the first usage of ND4J with the CUDA Backend (Windows)

```
Exception in thread "main" java.lang.RuntimeException: Can't allocate [HOST] memory: 32
```

If the Exception you are getting looks anything like this, and you see this upon startup:
```
o.n.j.c.CudaEnvironment - Device [0]: Free: 0 Total memory: 0
```

Then you are most probably trying to use a mobile GPU (like 970**m**) and Optimus is trying to ruin the day. First you should try to force the usage of the GPU through normal means, like setting the the JVM to run on your GPU via the Nvidia System Panel or by disabling the iGPU in your BIOS. If this still isn't enough, you can try the following workaround, that while **not recommended** for production, should allow you to still use your GPU.

You will have to add JOGL to your dependencies:
```xml
    <dependency>
      <groupId>org.jogamp.gluegen</groupId>
      <artifactId>gluegen-rt-main</artifactId>
      <version>2.3.1</version>
    </dependency>
    <dependency>
      <groupId>org.jogamp.jogl</groupId>
      <artifactId>jogl-all-main</artifactId>
      <version>2.3.1</version>
    </dependency>
```

And as the very first thing in your `main` method you will need to add:

```java
        GLProfile.initSingleton();
```

This should allow ND4J to work correctly (you still have to set that the JVM has to use the GPU in the Nvidia System Panel).


### My Display Driver / System crashes when I use the CUDA Backend (Windows)

ND4J is meant to be used with pure compute cards (i.e. the Tesla series). On consumer GPUs that are mainly meant for gaming, this results in a usage that can conflict with with the cards primary work: Displaying your Desktop. 

Microsoft has added the Timeout Detection and Recovery (TDR) to detect malfunctioning drivers and improper usage, which now interferes with the compute tasks of ND4J, by killing them if they occupy the GPU for longer then a few seconds. This results in the "Display driver stopped responding and has recovered" message. This results in a perceived driver crash along with a crash of your application. If you try to run it again TDR may decide that something is messing with the display driver and force a reboot.

If you really want to use your display GPU for compute with ND4J (**not recommended**), you will have to disable TDR by setting TdrLevel=0 (see https://msdn.microsoft.com/en-us/library/windows/hardware/ff569918%28v=vs.85%29.aspx). If you do this you **will** have display freezes, which, depending on your workload, can stay quite a long time.


### My JVM is crashing with the problematic frame being in `cygwin1.dll`

If you have any cygwin related dlls in the crash log, this means that you have build libnd4j or nd4j with cygwin being on the PATH before Msys2. This results in successful compilation, but crashes the JVM with some usecases. 

In order to fix this problem, all you have to do is to remove cygwin from your PATH while building libnd4j and nd4j.

If you want to inspect your path you can do this by running:
```
    echo $PATH
```

If you want to set your PATH temporarily, you can do so with:
```
    export PATH=... # Replace ... with what ever you want to have there
```

### CUDA build is failing with cmake/nmake errors

Some errors such as the following can appear if the visual studio vcvars64.bat file is run before attempting the cuda build.

```
  The parameter is incorrectRC Pass 1 failed to run.
  NMAKE : fatal error U1077: 'C:\msys64\mingw64\bin\cmake.exe' : return code '0xffffffff'
  NMAKE : fatal error U1077: '"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\amd64\nmake.exe"' : return code '0x2'
```

To resolve this, ensure that you haven't run vcvars64/vcvarsall in the msys2 shell before building.

#MSI Installer

To build an MSI Installer run:
./buildnativeoperations.sh -p msi

For gpu run:
./buildnativeoperations.sh -p msi -c cuda


#BLAS Impls

Openblas: Ensure that you set up $MSYSROOT/opt/OpenBLAS/lib. If you built OpenBLAS in msys2 (make, make install), then you should not need to do anything else.

Note: our informal/unscientific testing suggests that Intel MKL can be about equal with, and up to about 40% faster than OpenBLAS on some matrix multiply (gemm) operations, on some machines. Installing MKL is recommended but not required.

### MKL Setup

To build libnd4j with MKL:

- Download MKL from [https://software.intel.com/en-us/articles/free_mkl](https://software.intel.com/en-us/articles/free_mkl) and install. Registration is required (free).
- Add the \redist\intel64_win\mkl\ directory to your system PATH environment variable. This will be in a location such as C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2016.3.207\windows\redist\intel64_win\mkl\

Then build libnd4j as before. You may have to be careful about having multiple BLAS implementations on your path. Ideally, have only MKL on the path while building libnd4j.

Note: you may be able to get some additional performance on hyperthreaded processors by setting the system/environment variable MKL_DYNAMIC to have the value 'false'.
