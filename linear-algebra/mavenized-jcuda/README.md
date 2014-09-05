Mavenized JCuda
============

Features:

* Currently support _JCuda 0.5.0, 0.5.5, 0.6.0RC_ (windows-i386, windows-x86_64 only), 0.6.0 (windows-i386, windows-x86_64, unix-x86_64, mac-x86_64 only)
* Local Maven repo with 5 sets of native libs (windows-i386, windows-x86_64, unix-i386, unix-x86_64, mac-x86_64)
* Auto detection of OS family name and it's architecture (it's choose needed dependencies automatically)
* Running any main class, that contained JCuda code easily (without classpath hell, just run Maven goal)

How to run JCuda code:

* You need to install Cuda (5.0, 5.5, 6.0) for your platform [here] [1]
* Set property _<jcuda.version>_ in pom.xml to what you really use, e.g. 0.5.0 for Cuda 5.0, etc.
    * If you run **JCuda 0.6.0RC** and higher on **Windows** platform everything is ok
    * If you run **JCuda 0.5.5** and lower on **Windows** or **Unix** platform - you should set profile manually by adding *-P windows-x86_64_old*
        or *-P windows-x86_old* or *-P unix-x86_64_old*, etc. to _mvn clean package_ command, depends on architecture you use.
        For more info about **Maven** profiles take a look [here] [2]
    * If you run **JCuda 0.6.0** on **Unix x86_64** platform - everything is ok
    * If you run **JCUda 0.6.0** on **Mac x86_64** - everything is ok
* Call _mvn clean package_ to build project
* Call _mvn exec:exec_ to run main class (org.mystic.cuda.JCudaRuntimeTest) with "Hello, JCuda" sample :)
* ???
* Fork! Write your own JCuda code! Run! Report bugs! Support!

[Support me!]

[1]: https://developer.nvidia.com/cuda-downloads "here"
[2]: http://maven.apache.org/guides/introduction/introduction-to-profiles.html "here"
[Support me!]: https://www.gittip.com/MysterionRise/
