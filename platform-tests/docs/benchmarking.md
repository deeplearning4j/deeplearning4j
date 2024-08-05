> # Benchmarking CUDA with NCU in Deeplearning4j's platform-tests Module
>
> ## Introduction
>
> This document provides instructions for using NVIDIA's NCU (NVIDIA Compute Profiler) to benchmark CUDA performance in the `platform-tests` module of Deeplearning4j. It highlights the necessity of running the profiler as the root user and explains the rationale behind using a standalone JUnit console launcher instead of Maven Surefire.
>
> ## Why Run as Root?
>
> Running the NCU profiler as a root user is essential due to the elevated permissions required to load specific kernel modules necessary for CUDA profiling. Normal users may encounter permissions issues, hindering access to required system resources for kernel-level operations. Root access ensures unrestricted profiling capabilities.
>
> ## Why Standalone JUnit Console Instead of Maven Surefire?
>
> **Issues with Maven Surefire**:
>
> - **Freezing During Execution**: When using Maven Surefire for test execution, there have been observed instances of freezing, particularly after a number of test attempts. This issue becomes more pronounced with longer tests, potentially affecting the reliability and efficiency of the profiling process.
> - **Exacerbated by Long Tests**: Longer tests are more susceptible to these freezing issues. This inconsistency can lead to incomplete or unreliable profiling data, which is detrimental to the purpose of performance benchmarking.
>
> **Advantages of Standalone JUnit Console**:
>
> - **Stability**: The standalone JUnit console offers more stable execution of tests, especially those with longer durations, thereby providing more consistent and reliable profiling results.
> - **Flexibility**: It allows for greater control and flexibility in test execution, which is essential when profiling specific parts of the code, particularly in a CUDA environment.
> - **Test Modification**: For profiling purposes, tests might need temporary modifications to cater to profiling requirements. The standalone console facilitates these modifications more seamlessly than Maven Surefire.
>
> ## Pre-requisites
>
> - NVIDIA's NCU installed as part of the NVIDIA NSight Compute package.
> - The standalone JUnit console launcher.
> - A shaded (uber) jar of the Deeplearning4j `platform-tests` module.
>
> ## Setup
>
> 1. **Install NVIDIA NSight Compute**: Ensure that NVIDIA NSight Compute is installed on your system. It includes the NCU tool.
>
> 2. **Download JUnit Console Standalone**: Obtain the standalone JUnit console launcher from [Maven Central](https://repo1.maven.org/maven2/org/junit/platform/junit-platform-console-standalone/1.9.3/).
>
> 3. **Build Deeplearning4j `platform-tests` Uber Jar**: Compile the `platform-tests` module of Deeplearning4j into an uber jar.
>
> ## Running the Benchmark
>
> To run the benchmark, use the following command:
>
> ```
> /usr/local/cuda-12.1/nsight-compute-2023.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu \
> --config-file off \
> --export /root/profiler-output7.txt \
> --force-overwrite \
> --target-processes all \
> --replay-mode application \
> --app-replay-match all \
> --app-replay-buffer file \
> --app-replay-mode strict \
> --set detailed \
> --sampling-max-passes 1 \
> --check-exit-code no \
> java -cp /home/agibsonccc/Downloads/junit-platform-console-standalone-1.9.3.jar \
> org.junit.platform.console.ConsoleLauncher \
> -cp=target/platform-tests-1.0.0-SNAPSHOT-shaded.jar \
> -m=org.eclipse.deeplearning4j.dl4jcore.gradientcheck.CNN1DGradientCheckTest#testCnn1dWithMasking
> ```
>
> ## Understanding NCU Command Flags
>
> - **--config-file off**: Disables the use of default configuration file for profiling.
> - **--export /root/profiler-output7.txt**: Specifies the file path for saving the profiler's output.
> - **--force-overwrite**: Allows overwriting of the output file if it already exists.
> - **--target-processes all**: Targets all processes for profiling.
> - **--replay-mode application**: Sets the profiler to replay the entire application.
> - **--app-replay-match all**: Captures all instances of the application for replay.
> - **--app-replay-buffer file**: Uses file buffering for application replay.
> - **--app-replay-mode strict**: Enforces strict replay of the application.
> - **--set detailed**: Enables detailed profiling.
> - **--sampling-max-passes 1**: Limits the maximum number of sampling passes to one. This is crucial for reducing the profiling overhead and is particularly useful in scenarios where a lower overhead is desired or when profiling longer running kernels.
> - **--check-exit-code no**: Ignores the application's exit code during profiling.
>
> ## Important Notes
>
> - **Running as Root**: Necessary for loading kernel modules required for CUDA profiling.
> - **Avoiding Maven Surefire**: Due to stability issues with longer tests, the standalone JUnit console is preferred.
> - **Profiler Output**: Save to `/root/profiler-output7.txt`, ensuring accessibility and writability.
> - **Classpath Configuration**: Includes both the standalone JUnit console and the uber jar of the tests.
> - **Test Selection and Modification**: Adjust the module and test name as needed; modify tests temporarily for profiling if required.
>
> ## Conclusion
>
> Using NCU with the standalone JUnit console and a shaded jar of the `platform-tests` module enables comprehensive CUDA profiling in Deeplearning4j. This setup overcomes the limitations of Maven Surefire and permissions issues, ensuring a robust and reliable performance analysis.
