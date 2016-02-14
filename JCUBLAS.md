---
title: DL4J GPU Support
layout: default
---

# DL4J GPU Support

Deeplearning4j now includes basic GPU support. This means it can use your GPU device to enhance its performance.

Architecture
----
DL4J is written in Java and it communicated with the CUDA device via JCuda, a set of bindings for 
the CUDA toolkit. A copy of the jar files you can get from [the JCuda Project](http://jcuda.org/downloads/downloads.html)

On top of JCuda is the [JCublas](http://jcuda.org/jcuda/jcublas/JCublas.html) libraries. To quote:

    JCublas is a library that makes it it possible to use CUBLAS, the NVIDIA CUDA implementation of 
    the Basic Linear Algebra Subprograms, in Java applications. 

DL4J will use JCublas to greatly enhance mathematical calculation performance.



Quick Start Guide for Ubuntu 14
----

Make sure you have an NVidia device.

    lspci | grep -i nvidia
    
Since the version of JCuda we use is based on 5.5, we need to use 5.5 CUDA dev kit, too.

Uubntu 14 comes with CUDA 5.5 support already supported, so it's a matter of doing this

    sudo apt-get install libcudart5.5
    sudo apt-get install libcuda1-331
    
Now to make sure that it actually got installed.

    $ dpkg -l | grep nvidia
    rc  nvidia-304                     304.117-0ubuntu1       amd64        NVIDIA legacy binary driver - version 304.117
    ii  nvidia-331                     331.38-0ubuntu7.1      amd64        NVIDIA binary  driver - version 331.38
    ii  nvidia-331-dev                 331.38-0ubuntu7.1      amd64        NVIDIA binary Xorg driver development files
    rc  nvidia-331-updates             331.38-0ubuntu7.1      amd64        NVIDIA binary driver - version 331.38
    rc  nvidia-cuda-toolkit            5.5.22-3ubuntu1        amd64        NVIDIA CUDA toolkit
    ii  nvidia-libopencl1-331          331.38-0ubuntu7.1      amd64        NVIDIA OpenCL Driver and ICD Loader library
    rc  nvidia-libopencl1-331-updates  331.38-0ubuntu7.1      amd64        NVIDIA OpenCL Driver and ICD Loader library
    rc  nvidia-opencl-icd-304          304.117-0ubuntu1       amd64        NVIDIA OpenCL ICD
    ii  nvidia-opencl-icd-331          331.38-0ubuntu7.1      amd64        NVIDIA OpenCL ICD
    rc  nvidia-opencl-icd-331-updates  331.38-0ubuntu7.1      amd64        NVIDIA OpenCL ICD
    ii  nvidia-prime                   0.6.2                  amd64        Tools to enable NVIDIAs Prime
    ii  nvidia-settings                331.62-0ubuntu1        amd64        Tool for configuring the NVIDIA graphics driver

I also ended up having to install the CUDA libraries from NVidia as well. For older releases, of which 5.5 is, you need
to go to the [Legacy Toolkit Download](https://developer.nvidia.com/cuda-toolkit-archive) page.

We need the [CUDA 5.5 Toolkit](https://developer.nvidia.com/cuda-toolkit-55-archive). Download and install it. Grab the one 
for Ubuntu 12 if you have 14.

You should be good to go after this.

Testing it out
---

First, grab the JAR files from the JCuda project for CUDA 5.5 devkit. For us, we want [JCuda-All-0.5.5-bin-linux-x86_64.zip](http://www.jcuda.org/downloads/JCuda-All-0.5.5-bin-linux-x86_64.zip).

Unziping it should create a directory with these inside:
```
-rw-rw-r-- 1 user user 172345 Sep 13  2013 jcublas-0.5.5.jar
-rw-rw-r-- 1 user user 966392 Sep 13  2013 jcuda-0.5.5.jar
-rw-rw-r-- 1 user user  63441 Sep 13  2013 jcufft-0.5.5.jar
-rw-rw-r-- 1 user user  67325 Sep 13  2013 jcurand-0.5.5.jar
-rw-rw-r-- 1 user user 161388 Sep 13  2013 jcusparse-0.5.5.jar
-rw-rw-r-- 1 user user 322503 Oct  5  2013 libJCublas2-linux-x86_64.so
-rw-rw-r-- 1 user user 194291 Oct  5  2013 libJCublas-linux-x86_64.so
-rw-rw-r-- 1 user user 234380 Oct  5  2013 libJCudaDriver-linux-x86_64.so
-rw-rw-r-- 1 user user 204742 Oct  5  2013 libJCudaRuntime-linux-x86_64.so
-rw-rw-r-- 1 user user  71337 Oct  5  2013 libJCufft-linux-x86_64.so
-rw-rw-r-- 1 user user  72095 Oct  5  2013 libJCurand-linux-x86_64.so
-rw-rw-r-- 1 user user 329836 Oct  5  2013 libJCusparse2-linux-x86_64.so
-rw-rw-r-- 1 user user 157407 Oct  5  2013 libJCusparse-linux-x86_64.so
```

These .so files are critical to getting JCuda to run on your machine. I copied them into **/lib**.

    $ cp *.so /lib
    $ ldconfig

Take this file I took from the JCublas project:


```
/*
 * JCublas - Java bindings for CUBLAS, the NVIDIA CUDA BLAS library,
 * to be used with JCuda <br />
 * http://www.jcuda.org
 *
 * Copyright 2009 Marco Hutter - http://www.jcuda.org
 */

import java.util.Random;

import jcuda.*;
import jcuda.jcublas.JCublas;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.runtime.JCuda;

import static jcuda.jcublas.JCublas2.*;
import static jcuda.jcublas.cublasPointerMode.CUBLAS_POINTER_MODE_DEVICE;
import static jcuda.jcublas.cublasPointerMode.CUBLAS_POINTER_MODE_HOST;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;
/**
 * This is a sample class demonstrating the application of JCublas for
 * performing a BLAS 'sgemm' operation, i.e. for computing the matrix <br />
 * C = alpha * A * B + beta * C <br />
 * for single-precision floating point values alpha and beta, and matrices A, B
 * and C of size 1000x1000.
 */
public class JCublasSample
{
    public static void main(String args[])
    {
	//        testSgemm(10);
	cols(10);
    }

    public static void cols(int n) 
    {
        float alpha = 1f;
        float beta = 0.0f;

	int a_rows = n;
	int a_cols = 10;
	int b_rows = 10;
	int b_cols = n;

        JCublas2.setExceptionsEnabled(true);
        JCuda.setExceptionsEnabled(true);

        System.out.println("Creating input data...");
        float h_A[] = createRandomFloatData(a_rows*a_cols,1);
        float h_B[] = createRandomFloatData(a_rows*a_cols,2);
        float h_C[] = createRandomFloatData(a_rows*b_cols,1);
        float h_C_ref[] = h_C.clone();

        System.out.println("Performing Sgemm with Java...");
        sgemmJava(a_rows, a_cols, b_cols, alpha, h_A, h_B, beta, h_C_ref);

        System.out.println("Performing Sgemm with JCublas...");
        sgemmJCublas(a_rows, a_cols, b_cols, alpha, h_A, h_B, beta, h_C);

        boolean passed = isCorrectResult(h_C, h_C_ref);
	System.out.println("cols "+(passed?"PASSED":"FAILED"));
        System.out.println(h_C_ref);
        System.out.println(h_C);
        System.out.println("asdf");
        System.out.println(JCublas.cublasGetError());

    }
    /**
     * Test the JCublas sgemm operation for matrices of size n x x
     *
     * @param n The matrix size
     */
    public static void testSgemm(int n)
    {
        float alpha = 1f;
        float beta = 0.0f;

	int rows = n;
	int columns = n + 1;

        JCublas2.setExceptionsEnabled(true);
        JCuda.setExceptionsEnabled(true);

        System.out.println("Creating input data...");
        float h_A[] = createRandomFloatData(rows*rows,1);
        float h_B[] = createRandomFloatData(rows*columns,2);
        float h_C[] = createRandomFloatData(rows*rows,1);
        float h_C_ref[] = h_C.clone();

        System.out.println("Performing Sgemm with Java...");
        sgemmJava(rows, columns, rows, alpha, h_A, h_B, beta, h_C_ref);

        System.out.println("Performing Sgemm with JCublas...");
        sgemmJCublas(rows, columns, rows, alpha, h_A, h_B, beta, h_C);

        boolean passed = isCorrectResult(h_C, h_C_ref);
  	  System.out.println("testSgemm "+(passed?"PASSED":"FAILED"));
    }

    /**
     * Implementation of sgemm using JCublas
     */
    private static void sgemmJCublas(int m, int k, int n, float alpha, float A[], float B[],
				     float beta, float C[])
    {

        // Initialize JCublas
        JCublas.cublasInit();

        // Allocate memory on the device
        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();
        Pointer d_C = new Pointer();
        JCublas.cublasAlloc(m*k, Sizeof.FLOAT, d_A);
        JCublas.cublasAlloc(k*n, Sizeof.FLOAT, d_B);
        JCublas.cublasAlloc(m*n, Sizeof.FLOAT, d_C);

        // Copy the memory from the host to the device
        JCublas.cublasSetVector(m*k, Sizeof.FLOAT, Pointer.to(A), 1, d_A, 1);
        JCublas.cublasSetVector(k*n, Sizeof.FLOAT, Pointer.to(B), 1, d_B, 1);
        JCublas.cublasSetVector(m*n, Sizeof.FLOAT, Pointer.to(C), 1, d_C, 1);

	int lda = m;
	int ldb = k;
	int ldc = m;

        // Execute sgemm
        JCublas.cublasSgemm(
            'n', 
	    'n',
	    m, 
	    n,
	    k,
	    alpha,
	    d_A,
	    lda, 
	    d_B,
	    ldb, 
	    beta, 
	    d_C, 
	    ldc
	);

        // Copy the result from the device to the host
        JCublas.cublasGetVector(m*n, Sizeof.FLOAT, d_C, 1, Pointer.to(C), 1);

        // Clean up
        JCublas.cublasFree(d_A);
        JCublas.cublasFree(d_B);
        JCublas.cublasFree(d_C);
        JCublas.cublasShutdown();
    }

    /**
     * Simple implementation of sgemm, using plain Java
     */
    private static void sgemmJava(int n, int m, int z, float alpha, float A[], float B[],
				  float beta, float C[])
    {
	try {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < m; ++j)
            {
                float prod = 0;
                for (int k = 0; k < n; ++k)
                {
                    prod += A[k * n + i] * B[j * m + k];
                }
                C[j * n + i] = alpha * prod + beta * C[j * n + i];
            }
        }
	}
	catch (Exception e) {
	    System.err.println("Caught out of bounds: " + e.getMessage());

	}
	printMatr(A);
	printMatr(B);
	printMatr(C);

        System.out.println(JCublas.cublasGetError());

    }

    public static void printMatr(float C[]) {
        for (int i = 0; i < C.length; i++) {System.out.print(" " + Double.toString(C[i]));}	
	System.out.print("\n");
    }

    /**
     * Creates an array of the specified size, containing some random data
     */
    private static float[] createRandomFloatData(int n,float val)
    {
        Random random = new Random();
        float x[] = new float[n];
        for (int i = 0; i < n; i++)
        {
            x[i] = random.nextFloat();
	    x[i] = val;
        }
        return x;
    }

    /**
     * Compares the given result against a reference, and returns whether the
     * error norm is below a small epsilon threshold
     */
    private static boolean isCorrectResult(float result[], float reference[])
    {
        float errorNorm = 0;
        float refNorm = 0;
        for (int i = 0; i < result.length; ++i)
        {
            float diff = reference[i] - result[i];
            errorNorm += diff * diff;
            refNorm += reference[i] * result[i];
        }
        errorNorm = (float) Math.sqrt(errorNorm);
        refNorm = (float) Math.sqrt(refNorm);
        if (Math.abs(refNorm) < 1e-6)
        {
            return false;
        }
        return (errorNorm / refNorm < 1e-6f);
    }
}```

Now, go ahead and compile and run it. For simplicity's sake, I copy the jar files into the $CWD.

    $ cp JCuda-All-0.5.5-bin-linux-x86_64/*jar ./
    $ javac -cp ".:*" JCublasSample.java
    $ java -cp ".:*" JCublasSample
    Creating input data...
    Performing Sgemm with Java...
    Performing Sgemm with JCublas...
    PASSED
    
Hopefully you'll see the above.
