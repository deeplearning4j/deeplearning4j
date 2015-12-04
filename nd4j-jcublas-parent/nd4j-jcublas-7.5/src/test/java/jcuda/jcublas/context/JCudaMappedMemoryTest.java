package jcuda.jcublas.context;


import java.nio.*;

import jcuda.*;
import static jcuda.runtime.JCuda.*;
import static jcuda.jcublas.JCublas.*;

import jcuda.jcublas.JCublas;
import jcuda.runtime.*;

public class JCudaMappedMemoryTest
{
    public static void main(String args[])
    {
        // Enable exceptions to quickly be informed about errors in this test
        JCuda.setExceptionsEnabled(true);

        // Check if the device supports mapped host memory
        cudaDeviceProp deviceProperties = new cudaDeviceProp();
        cudaGetDeviceProperties(deviceProperties, 0);
        if (deviceProperties.canMapHostMemory == 0)
        {
            System.err.println("This device can not map host memory");
            System.err.println(deviceProperties.toFormattedString());
            return;
        }

        // Set the flag indicating that mapped memory will be used
        cudaSetDeviceFlags(cudaDeviceMapHost);

        // Allocate mappable host memory
        int n = 5;
        Pointer host = new Pointer();
        cudaHostAlloc(host, n * Sizeof.INT, cudaHostAllocMapped);

        // Create a device pointer mapping the host memory
        Pointer device = new Pointer();
        cudaHostGetDevicePointer(device, host, 0);

        // Obtain a ByteBuffer for accessing the data in the host
        // pointer. Modifications in this ByteBuffer will be
        // visible in the device memory.
        ByteBuffer byteBuffer = host.getByteBuffer(0, n * Sizeof.INT);

        // Set the byte order of the ByteBuffer
        byteBuffer.order(ByteOrder.nativeOrder());

        // For convenience, view the ByteBuffer as a FloatBuffer
        // and fill it with some sample data
        FloatBuffer floatBuffer = byteBuffer.asFloatBuffer();
        System.out.print("Input : ");
        for (int i = 0; i < n; i++)
        {
            floatBuffer.put(i, i);
            System.out.print(floatBuffer.get(i) + ", ");
        }

        JCublas.printVector(n,device);
        System.out.println();

      /*  // Apply a CUBLAS routine to the device pointer. This will
        // modify the host data, which was mapped to the device.
        cublasInit();
        cublasSscal(n, 2.0f, device, 1);
        cudaDeviceSynchronize();

        // Print the contents of the host memory after the
        // modification via the mapped pointer.
        System.out.print("Output: ");
        for (int i = 0; i < n; i++)
        {
            System.out.print(floatBuffer.get(i) + ", ");
        }
        System.out.println();*/

        // Clean up
        cudaFreeHost(host);
    }
}