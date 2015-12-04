package jcuda.jcublas.context;

/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 * http://www.jcuda.org
 *
 * Copyright 2013 Marco Hutter - http://www.jcuda.org
 */
import static jcuda.runtime.JCuda.cudaDeviceSynchronize;
import static jcuda.runtime.JCuda.cudaEventCreate;
import static jcuda.runtime.JCuda.cudaEventDestroy;
import static jcuda.runtime.JCuda.cudaEventElapsedTime;
import static jcuda.runtime.JCuda.cudaEventRecord;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaFreeHost;
import static jcuda.runtime.JCuda.cudaHostAlloc;
import static jcuda.runtime.JCuda.cudaHostAllocWriteCombined;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpyAsync;
import static jcuda.runtime.JCuda.cudaSetDevice;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

import java.nio.ByteBuffer;
import java.util.Locale;

import jcuda.Pointer;
import jcuda.runtime.cudaEvent_t;

/**
 * A test for the bandwidth of various copying operations.
 *
 * This test computes the bandwidth of the data transfer from the host to
 * the device. The host data is once allocated as pinned memory
 * (using cudaHostAlloc) and once as pageable memory (a Java array or
 * a direct buffer, comparable to 'malloc' in C).
 */
public class JCudaBandwidthTest
{
    /**
     * Memory modes for the host memory
     */
    enum HostMemoryMode
    {
        /**
         * Pinned host memory, allocated with cudaHostAlloc
         */
        PINNED,

        /**
         * Pageable memory in form of a Pointer.to(array)
         */
        PAGEABLE_ARRAY,

        /**
         * Pageable memory in form of a Pointer.to(directBuffer)
         */
        PAGEABLE_DIRECT_BUFFER,
    }

    /**
     * Entry point of this test
     *
     * @param args Not used
     */
    public static void main(String[] args)
    {
        int device = 0;
        cudaSetDevice(device);

        int hostAllocFlags = cudaHostAllocWriteCombined;
        runTest(HostMemoryMode.PINNED, hostAllocFlags);
        runTest(HostMemoryMode.PAGEABLE_ARRAY, hostAllocFlags);
        runTest(HostMemoryMode.PAGEABLE_DIRECT_BUFFER, hostAllocFlags);

        System.out.println("Done");
    }


    /**
     * Run a test that computes the bandwidth for copying host memory to the
     * device, using various memory block sizes, and print the results
     *
     * @param hostMemoryMode The {@link HostMemoryMode}
     * @param hostAllocFlags The flags for cudaHostAlloc
     */
    static void runTest(HostMemoryMode hostMemoryMode, int hostAllocFlags)
    {
        int minExponent = 10;
        int maxExponent = 28;
        int count = maxExponent - minExponent;
        int memorySizes[] = new int[count];
        float bandwidths[] = new float[memorySizes.length];

        System.out.print("Running");
        for (int i=0; i<count; i++)
        {
            System.out.print(".");
            memorySizes[i] = (1 << minExponent + i);
            float bandwidth = computeBandwidth(
                    hostMemoryMode, hostAllocFlags, memorySizes[i]);
            bandwidths[i] = bandwidth;
        }
        System.out.println();

        System.out.println("Bandwidths for "+hostMemoryMode);
        for (int i=0; i<memorySizes.length; i++)
        {
            String s = String.format("%10d", memorySizes[i]);
            String b = String.format(Locale.ENGLISH, "%5.3f", bandwidths[i]);
            System.out.println(s+" bytes : "+b+" MB/s");
        }
        System.out.println("\n");
    }


    /**
     * Compute the bandwidth in MB per second for copying data from the
     * host to the device
     *
     * @param hostMemoryMode The {@link HostMemoryMode}
     * @param hostAllocFlags The flags for the cudaHostAlloc call
     * @param memorySizes The memory sizes, in bytes
     * @param bandwidths Will store the bandwidth, in MB per second
     */
    static void computeBandwidths(
            HostMemoryMode hostMemoryMode, int hostAllocFlags,
            int memorySizes[], float bandwidths[])
    {
        for (int i=0; i<memorySizes.length; i++)
        {
            int memorySize = memorySizes[i];
            float bandwidth = computeBandwidth(
                    hostMemoryMode, hostAllocFlags, memorySize);
            bandwidths[i] = bandwidth;
        }
    }

    /**
     * Compute the bandwidth in MB per second for copying data from the
     * host to the device
     *
     * @param hostMemoryMode The {@link HostMemoryMode}
     * @param hostAllocFlags The flags for the cudaHostAlloc call
     * @param memorySize The memory size, in bytes
     * @return The bandwidth, in MB per second
     */
    static float computeBandwidth(
            HostMemoryMode hostMemoryMode, int hostAllocFlags, int memorySize)
    {
        // Initialize the host memory
        Pointer hostData = null;
        ByteBuffer hostDataBuffer = null;
        if (hostMemoryMode == HostMemoryMode.PINNED)
        {
            // Allocate pinned (page-locked) host memory
            hostData = new Pointer();
            cudaHostAlloc(hostData, memorySize, hostAllocFlags);
            hostDataBuffer = hostData.getByteBuffer(0, memorySize);
        }
        else if (hostMemoryMode == HostMemoryMode.PAGEABLE_ARRAY)
        {
            // The host memory is pageable and stored in a Java array
            byte array[] = new byte[memorySize];
            hostDataBuffer = ByteBuffer.wrap(array);
            hostData = Pointer.to(array);
        }
        else
        {
            // The host memory is pageable and stored in a direct byte buffer
            hostDataBuffer = ByteBuffer.allocateDirect(memorySize);
            hostData = Pointer.to(hostDataBuffer);
        }

        // Fill the memory with arbitrary data
        for (int i = 0; i < memorySize; i++)
        {
            hostDataBuffer.put(i, (byte)i);
        }

        // Allocate device memory
        Pointer deviceData = new Pointer();
        cudaMalloc(deviceData, memorySize);

        final int runs = 10;
        float bandwidth = computeBandwidth(
                deviceData, hostData, cudaMemcpyHostToDevice, memorySize, runs);

        // Clean up
        if (hostMemoryMode == HostMemoryMode.PINNED)
        {
            cudaFreeHost(hostData);
        }
        cudaFree(deviceData);
        return bandwidth;
    }


    /**
     * Compute the bandwidth in MB per second for copying data from the
     * given source pointer to the given destination pointer
     *
     * @param dstData The destination pointer
     * @param srcData The source pointer
     * @param memcopyKind The cudaMemcpyKind. Must match the types
     * of the source and destination pointers!
     * @param memSize The memory size, in bytes
     * @param runs The number of times that the copying operation
     * should be repeated
     * @return The bandwidth in MB per second
     */
    static float computeBandwidth(
            Pointer dstData, Pointer srcData,
            int memcopyKind, int memSize, int runs)
    {
        // Initialize the events for the time measure
        cudaEvent_t start = new cudaEvent_t();
        cudaEvent_t stop = new cudaEvent_t();
        cudaEventCreate(start);
        cudaEventCreate(stop);

        // Perform the specified number of copying operations
        cudaEventRecord(start, null);
        for (int i = 0; i < runs; i++)
        {
            cudaMemcpyAsync(dstData, srcData, memSize, memcopyKind, null);
        }
        cudaEventRecord(stop, null);
        cudaDeviceSynchronize();

        // Compute the elapsed time and bandwidth
        // in MB per second
        float elapsedTimeMsArray[] = { Float.NaN };
        cudaEventElapsedTime(elapsedTimeMsArray, start, stop);
        float elapsedTimeMs = elapsedTimeMsArray[0];
        float bandwidthInBytesPerMs =
                ((float) memSize * runs) / elapsedTimeMs;
        float bandwidth = bandwidthInBytesPerMs / 1024;

        // Clean up
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        return bandwidth;
    }
}