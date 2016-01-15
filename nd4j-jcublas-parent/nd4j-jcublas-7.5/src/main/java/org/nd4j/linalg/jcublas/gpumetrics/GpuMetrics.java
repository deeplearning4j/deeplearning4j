package org.nd4j.linalg.jcublas.gpumetrics;

import jcuda.Sizeof;
import jcuda.driver.CUoccupancyB2DSize;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.cudaDeviceProp;
import jcuda.utils.KernelLauncher;
import lombok.AllArgsConstructor;
import lombok.Data;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.nd4j.linalg.jcublas.kernel.KernelFunctionLoader;

import static  jcuda.runtime.JCuda.*;
import org.nd4j.linalg.jcublas.util.PointerUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 See:
 http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
 *
 * @author Adam Gibson
 */
@Data
@AllArgsConstructor
public class GpuMetrics  {
    public GpuMetrics() {
    }

    private static Logger log = LoggerFactory.getLogger(GpuMetrics.class);
    public final static int MAX_THREADS = 256;
    public final static int MAX_BLOCKS = 64;
    private int gridSize,blockSize,sharedMemory;
    private static  CUoccupancyB2DSize DOUBLE = new CUoccupancyB2DSize() {

        @Override
        public long call(int blockSize) {
            return blockSize * Sizeof.DOUBLE;
        }
    };

    private static CUoccupancyB2DSize FLOAT = new CUoccupancyB2DSize() {
        @Override
        public long call(int blockSize) {
            return blockSize * Sizeof.FLOAT;
        }
    };


    /**
     * Outputs the expected gpu information
     * to send to the gpu for cuda
     * kernel metadata.
     * The first entry is the block size
     * The second entry is the grid size
     * The third entry is the shared memory
     * @return a 3 length array
     * representing the gpu information
     */
    public int[] getGpuDefinitionInfo() {
        int[] gpuDef = new int[4];
        gpuDef[0] = getBlockSize();
        gpuDef[1] = getGridSize();
        gpuDef[2] = getSharedMemory();
        gpuDef[3] = ContextHolder.getInstance().getCurrentGpuInformation().getMaxSharedMemoryPerBlock();
        return gpuDef;
    }

    public int getGridSize() {
        return gridSize;
    }

    public int getBlockSize() {
        return blockSize;
    }

    public int getSharedMemory() {
        return sharedMemory;
    }

    /**
     * Given n, max threads
     * @param n the number of elements to process
     * @param maxThreads the max number of threads
     * @param maxBlocks the max number of blocks
     * @return an array with the number of threads as
     * the first entry and number of blocks
     * as the second entry
     */
    public static int[] getThreadsAndBlocks(int n,int maxThreads,int maxBlocks) {
        //get device capability, to avoid block/grid size exceed the upper bound
        cudaDeviceProp prop = new cudaDeviceProp();
        int[] devicePointer = new int[1];
        cudaGetDevice(devicePointer);
        cudaGetDeviceProperties(prop, devicePointer[0]);


        int threads = (n < maxThreads*2) ? PointerUtil.nextPow2((n + 1) / 2) : maxThreads;
        int blocks = (n + (threads * 2 - 1)) / (threads * 2);


        if ((float) threads * blocks > (float) prop.maxGridSize[0] * prop.maxThreadsPerBlock)
        {
            throw new IllegalStateException("n is too large, please choose a smaller number!\n");
        }

        if (blocks > prop.maxGridSize[0])
        {
            log.warn("Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n",
                    blocks, prop.maxGridSize[0], threads * 2, threads);

            blocks /= 2;
            threads *= 2;
        }


        blocks = Math.min(maxBlocks, blocks);
        return new int[] {threads,blocks};
    }


    /**
     * Get the blocks and threads
     * used for a kernel launch
     * @param dataType the data type
     * @param n the number of elements
     * @return the information used
     * for launching a kernel
     */
    public  static GpuMetrics blockAndThreads(DataBuffer.Type dataType,int n) {
        //<<<numBlocks, threadsPerBlock>>>
        //<<< gridSize, blockSize >>>
        int size = dataType.equals(DataBuffer.Type.DOUBLE) ? Sizeof.DOUBLE : Sizeof.FLOAT;
        int[] threadsAndBlocks = getThreadsAndBlocks(n,MAX_THREADS,MAX_BLOCKS);
        int sharedMemSize =   (threadsAndBlocks[0] <= 32) ? 2 * threadsAndBlocks[0] * size : threadsAndBlocks[0] * size;
        return new GpuMetrics(threadsAndBlocks[0],threadsAndBlocks[1],sharedMemSize);
    }


    /**
     *
     * @param functionName
     * @param dataType
     * @param n
     * @return
     */
    public static GpuMetrics blocksAndThreadsOccupancy(String functionName, DataBuffer.Type dataType, int n) {
        int[] gridSize = new int[1];
        int[] blockSize = new int[1];
        KernelLauncher launcher = KernelFunctionLoader.launcher(functionName, dataType);
        if (launcher == null) throw new IllegalStateException("KernelLauncher is null");
        CUoccupancyB2DSize size = dataType.equals(DataBuffer.Type.FLOAT) ? FLOAT : DOUBLE;
        JCudaDriver.cuOccupancyMaxPotentialBlockSize(gridSize,blockSize,launcher.getFunction(),size,0,0);

        int gridSizeRet = (n +  blockSize[0] - 1) / blockSize[0];
        int blockSizeRet  = blockSize[0];
        //for smaller problems, ensure no index out of bounds
        if(blockSizeRet > n)
            blockSizeRet = n;
        int maxBlockSize = ContextHolder.getInstance().getCurrentGpuInformation().getMaxThreadsPerBlock();
        if(blockSizeRet > maxBlockSize)
            blockSizeRet = maxBlockSize;
        int maxGridSize = ContextHolder.getInstance().getCurrentGpuInformation().getMaxGrimDimX();
        if(gridSizeRet > maxGridSize)
            gridSizeRet = maxGridSize;
        int maxSharedMem = ContextHolder.getInstance().getCurrentGpuInformation().getMaxSharedMemoryPerBlock();
        int sharedMemSize = blockSizeRet * (dataType.equals(DataBuffer.Type.FLOAT) ? Sizeof.FLOAT : Sizeof.DOUBLE);
        if(sharedMemSize > maxSharedMem)
            sharedMemSize = maxSharedMem;
        return new GpuMetrics(gridSizeRet,blockSizeRet,sharedMemSize);
    }


    /**
     * Validates the current configuration
     * against the gpu's hardware constraints.
     *
     * Throws an {@link IllegalArgumentException}
     * if any of the values surpass the GPU's
     * built in hardware constraints
     */
    public void validate() {
        int maxGrid = ContextHolder.getInstance().getCurrentGpuInformation().getMaxThreadsPerBlock();
        int maxBlock = ContextHolder.getInstance().getCurrentGpuInformation().getMaxBlockDimx();
        int maxShared = ContextHolder.getInstance().getCurrentGpuInformation().getMaxSharedMemoryPerBlock();
        if(gridSize > maxGrid)
            throw new IllegalArgumentException("Maximum grid size is " + maxGrid + " but was specified as " + gridSize);
        if(blockSize > maxBlock)
            throw new IllegalArgumentException("Maximum block size is " + maxBlock + " but was specified as " + blockSize);
        if(sharedMemory > maxShared)
            throw new IllegalArgumentException("Maximum shared memory size per block is " + maxShared + " but was specified as " + sharedMemory);
    }


    /**
     * Special setter that queries
     * the maximum amount of shared memory per block allowed
     * @param sharedMemory
     */
    public void setSharedMemoryNotOverMax(int sharedMemory) {
        setSharedMemory(Math.min(sharedMemory,1024));
    }

    /**
     * Special setter that queries
     * the maximum amount of shared memory per block allowed
     * @param gridSize
     */
    public void setGridSizeNotOverMax(int gridSize) {
        setGridSize(Math.min(gridSize,ContextHolder.getInstance().getCurrentGpuInformation().getMaxThreadsPerBlock()));
    }

    /**
     * Special setter
     * that queries the block size
     * to ensure not over the max possible
     * block size is specified.
     * @param blockSize the block size to attempt to set
     */
    public void setBlockSizeNotOverMax(int blockSize) {
        setBlockSize(Math.min(blockSize,ContextHolder.getInstance().getCurrentGpuInformation().getMaxBlockDimx()));
    }



}
