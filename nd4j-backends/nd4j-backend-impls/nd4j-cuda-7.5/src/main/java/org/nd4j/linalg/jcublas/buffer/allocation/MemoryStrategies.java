package org.nd4j.linalg.jcublas.buffer.allocation;


import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.nd4j.linalg.jcublas.context.GpuInformation;

import java.util.ArrayList;
import java.util.List;

import static jcuda.driver.CUdevice_attribute.*;

/**
 * Based on the jcuda memory bandwidth example:
 * http://www.jcuda.org/samples/JCudaBandwidthTest.java
 *
 * @author Adam Gibson
 */
@Deprecated
public class MemoryStrategies {
    /**
     * Memory modes for the host memory
     *  (Credit to Marco from jcuda)
     */
    public  enum MemoryMode {
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
     * Get the mode for the device
     * based on the device information
     * @param device the device to get the mode for
     * @return the default mode for the device
     * based on the device characteristics
     */
    public static MemoryMode getMode(int device) {
        GpuInformation info = ContextHolder.getInstance().getInfoFor(device);
        //pinned is preferred
        if(info.getCanMapHostMemory() > 0) {
            return MemoryMode.PINNED;
        }

        return MemoryMode.PAGEABLE_DIRECT_BUFFER;
    }


    /**
     * Get the strategy for a given mode
     * @param mode the mode to get the strategy for
     * @return the strategy for the given mode
     */
    public static MemoryStrategy getStrategy(MemoryMode mode) {
        switch(mode) {
            case PINNED: return new PinnedMemoryStrategy();
            case PAGEABLE_ARRAY: return new PageableArrayMemoryStrategy();
            case PAGEABLE_DIRECT_BUFFER: return new PageableDirectBufferMemoryStrategy();
            default: throw new IllegalStateException("Illegal strategy for mode " + mode);
        }

    }

    /**
     * Get the configured memory strategy for the device
     * @param device the device
     *               to get the memory strategy for
     * @return the memory strategy
     */
    public static MemoryStrategy getStrategy(int device) {
        return getStrategy(getMode(device));
    }

    /**
     * Returns a short description of the given CUdevice_attribute constant
     *
     * @param attribute The CUdevice_attribute constant
     * @return A short description of the given constant
     */
    private static String getAttributeDescription(int attribute) {
        switch (attribute) {
            case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
                return "Maximum number of threads per block";
            case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X:
                return "Maximum x-dimension of a block";
            case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y:
                return "Maximum y-dimension of a block";
            case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z:
                return "Maximum z-dimension of a block";
            case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X:
                return "Maximum x-dimension of a grid";
            case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y:
                return "Maximum y-dimension of a grid";
            case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z:
                return "Maximum z-dimension of a grid";
            case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK:
                return "Maximum shared memory per thread block in bytes";
            case CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY:
                return "Total constant memory on the device in bytes";
            case CU_DEVICE_ATTRIBUTE_WARP_SIZE:
                return "Warp size in threads";
            case CU_DEVICE_ATTRIBUTE_MAX_PITCH:
                return "Maximum pitch in bytes allowed for memory copies";
            case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK:
                return "Maximum number of 32-bit registers per thread block";
            case CU_DEVICE_ATTRIBUTE_CLOCK_RATE:
                return "Clock frequency in kilohertz";
            case CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT:
                return "Alignment requirement";
            case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT:
                return "Number of multiprocessors on the device";
            case CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT:
                return "Whether there is a run time limit on kernels";
            case CU_DEVICE_ATTRIBUTE_INTEGRATED:
                return "Device is integrated with host memory";
            case CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY:
                return "Device can map host memory into CUDA address space";
            case CU_DEVICE_ATTRIBUTE_COMPUTE_MODE:
                return "Compute mode";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH:
                return "Maximum 1D texture width";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH:
                return "Maximum 2D texture width";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT:
                return "Maximum 2D texture height";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH:
                return "Maximum 3D texture width";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT:
                return "Maximum 3D texture height";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH:
                return "Maximum 3D texture depth";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH:
                return "Maximum 2D layered texture width";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT:
                return "Maximum 2D layered texture height";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS:
                return "Maximum layers in a 2D layered texture";
            case CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT:
                return "Alignment requirement for surfaces";
            case CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS:
                return "Device can execute multiple kernels concurrently";
            case CU_DEVICE_ATTRIBUTE_ECC_ENABLED:
                return "Device has ECC support enabled";
            case CU_DEVICE_ATTRIBUTE_PCI_BUS_ID:
                return "PCI bus ID of the device";
            case CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID:
                return "PCI device ID of the device";
            case CU_DEVICE_ATTRIBUTE_TCC_DRIVER:
                return "Device is using TCC driver model";
            case CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE:
                return "Peak memory clock frequency in kilohertz";
            case CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH:
                return "Global memory bus width in bits";
            case CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE:
                return "Size of L2 cache in bytes";
            case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR:
                return "Maximum resident threads per multiprocessor";
            case CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT:
                return "Number of asynchronous engines";
            case CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING:
                return "Device shares a unified address space with the host";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH:
                return "Maximum 1D layered texture width";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS:
                return "Maximum layers in a 1D layered texture";
            case CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID:
                return "PCI domain ID of the device";
        }
        return "(UNKNOWN ATTRIBUTE)";
    }

    /**
     * Returns a list of all CUdevice_attribute constants
     *
     * @return A list of all CUdevice_attribute constants
     */
    private static List<Integer> getAttributes() {
        List<Integer> list = new ArrayList<>();
        list.add(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK);
        list.add(CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY);
        list.add(CU_DEVICE_ATTRIBUTE_WARP_SIZE);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_PITCH);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK);
        list.add(CU_DEVICE_ATTRIBUTE_CLOCK_RATE);
        list.add(CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT);
        list.add(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
        list.add(CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT);
        list.add(CU_DEVICE_ATTRIBUTE_INTEGRATED);
        list.add(CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY);
        list.add(CU_DEVICE_ATTRIBUTE_COMPUTE_MODE);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS);
        list.add(CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT);
        list.add(CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS);
        list.add(CU_DEVICE_ATTRIBUTE_ECC_ENABLED);
        list.add(CU_DEVICE_ATTRIBUTE_PCI_BUS_ID);
        list.add(CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID);
        list.add(CU_DEVICE_ATTRIBUTE_TCC_DRIVER);
        list.add(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE);
        list.add(CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH);
        list.add(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR);
        list.add(CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT);
        list.add(CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS);
        list.add(CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID);
        return list;
    }

}
