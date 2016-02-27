package org.nd4j.linalg.jcublas.context;

import jcuda.driver.CUdevice;
import jcuda.driver.JCudaDriver;

import java.io.Serializable;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;

import static jcuda.driver.CUdevice_attribute.*;


/**
 * @author Adam Gibson
 */
public class GpuInformation implements Serializable {
    private int maxThreadsPerBlock;
    private int maxBlockDimx;
    private int maxBlockDimY;
    private int maxBlockDimZ;
    private int maxGrimDimX;
    private int maxGridDimY;
    private int maxGridDimZ;
    private int maxSharedMemoryPerBlock;
    private int totalConstantMemory;
    private int warpSize;
    private int maxPitch;
    private int maxRegistersPerBlock;
    private int clockRate;
    private int textureAlignment;
    private int multiProcessorCount;
    private int kernelExecTimeOut;
    private int attributeIntegrated;
    private int canMapHostMemory;
    private int computeMode;
    private int maxTexture1dwidth;
    private int maxTexture2dWidth;
    private int texture2dHeight;
    private int texture3dHeight;
    private int texture3dDepth;
    private int texture2dLayeredWidth;
    private int texture2dLayeredHeight;
    private int texture2DLayerLayers;
    private int attributeSurfaceAlignment;
    private int concurrentKernels;
    private int cuDeviceAttributeEccEnabled;
    private int picDeviceId;
    private int tccDriver;
    private int attributeMemoryClockRate;
    private int globalMemoryBusWidth;
    private int attributeL2CacheSize;
    private int maxThreadsPerMultiProcessor;
    private int asyncEngineCount;
    private int unifiedAddressing;
    private int maxTeture1dLayeredWidth;
    private int texture1dLayeredLayers;
    private int pciDomainId;

    /**
     * INitialize this information with the given device
     * @param device the device to initialize with
     */
    public GpuInformation(CUdevice device) {
        init(device);
    }

    /**
     * Initialize the device based on the device number
     * @param device
     */
    public GpuInformation(int device) {
        CUdevice cUdevice = new CUdevice();
        JCudaDriver.cuDeviceGet(cUdevice, device);
        init(cUdevice);
    }


    private void init(CUdevice device) {
        Field[] fields = getClass().getDeclaredFields();
        List<Integer> attributes = getAttributes();
        for(int i = 0; i < fields.length; i++) {
            int array[] = { 0 };
            JCudaDriver.cuDeviceGetAttribute(array, attributes.get(i), device);
            int value = array[0];
            try {
                fields[i].setAccessible(true);
                fields[i].set(this,value);
            } catch (IllegalAccessException e) {
                e.printStackTrace();
            }
        }
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

    public int getMaxThreadsPerBlock() {
        return maxThreadsPerBlock;
    }

    public void setMaxThreadsPerBlock(int maxThreadsPerBlock) {
        this.maxThreadsPerBlock = maxThreadsPerBlock;
    }

    public int getMaxBlockDimx() {
        return maxBlockDimx;
    }

    public void setMaxBlockDimx(int maxBlockDimx) {
        this.maxBlockDimx = maxBlockDimx;
    }

    public int getMaxBlockDimY() {
        return maxBlockDimY;
    }

    public void setMaxBlockDimY(int maxBlockDimY) {
        this.maxBlockDimY = maxBlockDimY;
    }

    public int getMaxBlockDimZ() {
        return maxBlockDimZ;
    }

    public void setMaxBlockDimZ(int maxBlockDimZ) {
        this.maxBlockDimZ = maxBlockDimZ;
    }

    public int getMaxGrimDimX() {
        return maxGrimDimX;
    }

    public void setMaxGrimDimX(int maxGrimDimX) {
        this.maxGrimDimX = maxGrimDimX;
    }

    public int getMaxGridDimY() {
        return maxGridDimY;
    }

    public void setMaxGridDimY(int maxGridDimY) {
        this.maxGridDimY = maxGridDimY;
    }

    public int getMaxGridDimZ() {
        return maxGridDimZ;
    }

    public void setMaxGridDimZ(int maxGridDimZ) {
        this.maxGridDimZ = maxGridDimZ;
    }

    public int getMaxSharedMemoryPerBlock() {
        return maxSharedMemoryPerBlock;
    }

    public void setMaxSharedMemoryPerBlock(int maxSharedMemoryPerBlock) {
        this.maxSharedMemoryPerBlock = maxSharedMemoryPerBlock;
    }

    public int getTotalConstantMemory() {
        return totalConstantMemory;
    }

    public void setTotalConstantMemory(int totalConstantMemory) {
        this.totalConstantMemory = totalConstantMemory;
    }

    public int getWarpSize() {
        return warpSize;
    }

    public void setWarpSize(int warpSize) {
        this.warpSize = warpSize;
    }

    public int getMaxPitch() {
        return maxPitch;
    }

    public void setMaxPitch(int maxPitch) {
        this.maxPitch = maxPitch;
    }

    public int getMaxRegistersPerBlock() {
        return maxRegistersPerBlock;
    }

    public void setMaxRegistersPerBlock(int maxRegistersPerBlock) {
        this.maxRegistersPerBlock = maxRegistersPerBlock;
    }

    public int getClockRate() {
        return clockRate;
    }

    public void setClockRate(int clockRate) {
        this.clockRate = clockRate;
    }

    public int getTextureAlignment() {
        return textureAlignment;
    }

    public void setTextureAlignment(int textureAlignment) {
        this.textureAlignment = textureAlignment;
    }

    public int getMultiProcessorCount() {
        return multiProcessorCount;
    }

    public void setMultiProcessorCount(int multiProcessorCount) {
        this.multiProcessorCount = multiProcessorCount;
    }

    public int getKernelExecTimeOut() {
        return kernelExecTimeOut;
    }

    public void setKernelExecTimeOut(int kernelExecTimeOut) {
        this.kernelExecTimeOut = kernelExecTimeOut;
    }

    public int getAttributeIntegrated() {
        return attributeIntegrated;
    }

    public void setAttributeIntegrated(int attributeIntegrated) {
        this.attributeIntegrated = attributeIntegrated;
    }

    public int getCanMapHostMemory() {
        return canMapHostMemory;
    }

    public void setCanMapHostMemory(int canMapHostMemory) {
        this.canMapHostMemory = canMapHostMemory;
    }

    public int getComputeMode() {
        return computeMode;
    }

    public void setComputeMode(int computeMode) {
        this.computeMode = computeMode;
    }

    public int getMaxTexture1dwidth() {
        return maxTexture1dwidth;
    }

    public void setMaxTexture1dwidth(int maxTexture1dwidth) {
        this.maxTexture1dwidth = maxTexture1dwidth;
    }

    public int getMaxTexture2dWidth() {
        return maxTexture2dWidth;
    }

    public void setMaxTexture2dWidth(int maxTexture2dWidth) {
        this.maxTexture2dWidth = maxTexture2dWidth;
    }

    public int getTexture2dHeight() {
        return texture2dHeight;
    }

    public void setTexture2dHeight(int texture2dHeight) {
        this.texture2dHeight = texture2dHeight;
    }

    public int getTexture3dHeight() {
        return texture3dHeight;
    }

    public void setTexture3dHeight(int texture3dHeight) {
        this.texture3dHeight = texture3dHeight;
    }

    public int getTexture3dDepth() {
        return texture3dDepth;
    }

    public void setTexture3dDepth(int texture3dDepth) {
        this.texture3dDepth = texture3dDepth;
    }

    public int getTexture2dLayeredWidth() {
        return texture2dLayeredWidth;
    }

    public void setTexture2dLayeredWidth(int texture2dLayeredWidth) {
        this.texture2dLayeredWidth = texture2dLayeredWidth;
    }

    public int getTexture2dLayeredHeight() {
        return texture2dLayeredHeight;
    }

    public void setTexture2dLayeredHeight(int texture2dLayeredHeight) {
        this.texture2dLayeredHeight = texture2dLayeredHeight;
    }

    public int getTexture2DLayerLayers() {
        return texture2DLayerLayers;
    }

    public void setTexture2DLayerLayers(int texture2DLayerLayers) {
        this.texture2DLayerLayers = texture2DLayerLayers;
    }

    public int getAttributeSurfaceAlignment() {
        return attributeSurfaceAlignment;
    }

    public void setAttributeSurfaceAlignment(int attributeSurfaceAlignment) {
        this.attributeSurfaceAlignment = attributeSurfaceAlignment;
    }

    public int getConcurrentKernels() {
        return concurrentKernels;
    }

    public void setConcurrentKernels(int concurrentKernels) {
        this.concurrentKernels = concurrentKernels;
    }

    public int getCuDeviceAttributeEccEnabled() {
        return cuDeviceAttributeEccEnabled;
    }

    public void setCuDeviceAttributeEccEnabled(int cuDeviceAttributeEccEnabled) {
        this.cuDeviceAttributeEccEnabled = cuDeviceAttributeEccEnabled;
    }

    public int getPicDeviceId() {
        return picDeviceId;
    }

    public void setPicDeviceId(int picDeviceId) {
        this.picDeviceId = picDeviceId;
    }

    public int getTccDriver() {
        return tccDriver;
    }

    public void setTccDriver(int tccDriver) {
        this.tccDriver = tccDriver;
    }

    public int getAttributeMemoryClockRate() {
        return attributeMemoryClockRate;
    }

    public void setAttributeMemoryClockRate(int attributeMemoryClockRate) {
        this.attributeMemoryClockRate = attributeMemoryClockRate;
    }

    public int getGlobalMemoryBusWidth() {
        return globalMemoryBusWidth;
    }

    public void setGlobalMemoryBusWidth(int globalMemoryBusWidth) {
        this.globalMemoryBusWidth = globalMemoryBusWidth;
    }

    public int getAttributeL2CacheSize() {
        return attributeL2CacheSize;
    }

    public void setAttributeL2CacheSize(int attributeL2CacheSize) {
        this.attributeL2CacheSize = attributeL2CacheSize;
    }

    public int getMaxThreadsPerMultiProcessor() {
        return maxThreadsPerMultiProcessor;
    }

    public void setMaxThreadsPerMultiProcessor(int maxThreadsPerMultiProcessor) {
        this.maxThreadsPerMultiProcessor = maxThreadsPerMultiProcessor;
    }

    public int getAsyncEngineCount() {
        return asyncEngineCount;
    }

    public void setAsyncEngineCount(int asyncEngineCount) {
        this.asyncEngineCount = asyncEngineCount;
    }

    public int getUnifiedAddressing() {
        return unifiedAddressing;
    }

    public void setUnifiedAddressing(int unifiedAddressing) {
        this.unifiedAddressing = unifiedAddressing;
    }

    public int getMaxTeture1dLayeredWidth() {
        return maxTeture1dLayeredWidth;
    }

    public void setMaxTeture1dLayeredWidth(int maxTeture1dLayeredWidth) {
        this.maxTeture1dLayeredWidth = maxTeture1dLayeredWidth;
    }

    public int getTexture1dLayeredLayers() {
        return texture1dLayeredLayers;
    }

    public void setTexture1dLayeredLayers(int texture1dLayeredLayers) {
        this.texture1dLayeredLayers = texture1dLayeredLayers;
    }

    public int getPciDomainId() {
        return pciDomainId;
    }

    public void setPciDomainId(int pciDomainId) {
        this.pciDomainId = pciDomainId;
    }
}
