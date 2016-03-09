/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.jcublas.context;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import jcuda.CudaException;
import jcuda.Pointer;
import jcuda.driver.*;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;
import jcuda.runtime.cudaStream_t;
import lombok.Data;
import org.apache.commons.pool2.impl.GenericObjectPoolConfig;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.allocation.MemoryStrategy;
import org.nd4j.linalg.jcublas.context.pool.CublasHandlePool;
import org.nd4j.linalg.jcublas.context.pool.OldStreamPool;
import org.nd4j.linalg.jcublas.context.pool.StreamPool;
import org.nd4j.linalg.jcublas.context.pool.factory.CublasHandlePooledItemFactory;
import org.nd4j.linalg.jcublas.context.pool.factory.OldStreamItemFactory;
import org.nd4j.linalg.jcublas.context.pool.factory.StreamItemFactory;
import org.nd4j.linalg.jcublas.device.conf.DeviceConfiguration;

import org.nd4j.linalg.jcublas.util.PointerUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.nd4j.linalg.io.ClassPathResource;

import org.apache.commons.pool2.ObjectPool;


import java.io.IOException;
import java.io.InputStream;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

import static jcuda.driver.JCudaDriver.*;

/**
 * A multithreaded version derived
 * from the cuda launcher util
 * by the authors of jcuda.
 *
 * This class handles managing cuda contexts
 * across multiple devices and threads.
 *
 *
 * @author Adam Gibson
 */
@Data
public class ContextHolder {

    private Map<Integer,CUdevice> devices = new ConcurrentHashMap<>();
    private Map<Integer,GpuInformation> info = new ConcurrentHashMap<>();
    private Table<Integer, String,CUcontext> deviceIDContexts = HashBasedTable.create();
    private Map<String,Integer> threadNameToDeviceNumber = new ConcurrentHashMap<>();
    private Table<CUcontext,String,CUstream> contextStreams = HashBasedTable.create();
    private Table<CUcontext,String,cudaStream_t> cudaStreams = HashBasedTable.create();
    private Map<String, cublasHandle> handleMap = new ConcurrentHashMap<>();
    private Map<String,Integer> threads = new ConcurrentHashMap<>();
    private List<Integer> bannedDevices;
    private int numDevices = 0;
    private Map<Integer,DeviceConfiguration> confs = new ConcurrentHashMap<>();
    private ObjectPool<cublasHandle> handlePool;
    private ObjectPool<CUstream> streamPool;
    private ObjectPool<cudaStream_t> oldStreamPool;
    private static ContextHolder INSTANCE;
    public final static String DEVICES_TO_BAN = "org.nd4j.linalg.jcuda.jcublas.ban_devices";
    private static AtomicBoolean deviceSetup = new AtomicBoolean(false);
    private boolean confCalled = false;
    private static Logger log = LoggerFactory.getLogger(ContextHolder.class);
    private AtomicBoolean shutdown = new AtomicBoolean(false);

    // holder for memory strategies override
    private Map<String, MemoryStrategy> forcedStrategies = new ConcurrentHashMap<>();

    /**
     * Singleton pattern
     * @return the instance for the context holder.
     */
    public static synchronized  ContextHolder getInstance() {

        if(INSTANCE == null) {
            Properties props = new Properties();
            try {
                props.load(new ClassPathResource("/cudafunctions.properties", ContextHolder.class.getClassLoader()).getInputStream());
            } catch (IOException e) {
                throw new RuntimeException(e);
            }

            INSTANCE = new ContextHolder();
            INSTANCE.configure();


            //set the properties to be accessible globally
            for(String pair : props.stringPropertyNames())
                System.getProperties().put(pair,props.getProperty(pair));




            Runtime.getRuntime().addShutdownHook(new Thread(new Runnable() {
                @Override
                public void run() {
                    INSTANCE.destroy();
                }
            }));
        }



        return INSTANCE;
    }

    public ObjectPool<cublasHandle> getHandlePool() {
        return handlePool;
    }

    public ObjectPool<CUstream> getStreamPool() {
        return streamPool;
    }

    public ObjectPool<cudaStream_t> getOldStreamPool() {
        return oldStreamPool;
    }

    public int getNumThreads(Op op) {
        String functionName = op instanceof TransformOp || op instanceof Accumulation ? op.name() + "_strided" : op.name();
        Integer threadsForFunction = ContextHolder.getInstance().getThreads().get(functionName);
        int maxThreads = ContextHolder.getInstance().getInfoFor(ContextHolder.getInstance().getDeviceForThread()).getMaxThreadsPerBlock();

        if(threadsForFunction == null)
            return PointerUtil.getNumThreads(op.n(), maxThreads);
        return threadsForFunction;

    }

    public Map<String, Integer> getThreads() {
        return threads;
    }

    /**
     * This methord forces use of specific MemoryStrategy for current thread
     *
     * PLEASE NOTE: NEVER USE THIS METHOD IN PRODUCTION ENVIRONMENT, IT CAN LEAD TO UNPREDICTABLE RESULTS
     *
     * @param memoryStrategy MemoryStrategy to be used withing current thread, if null - forced strategy for current thread will be purged
     */
    public void forceMemoryStrategyForThread(MemoryStrategy memoryStrategy) {
        if (memoryStrategy == null) forcedStrategies.remove(Thread.currentThread().getName());
            else forcedStrategies.put(Thread.currentThread().getName(), memoryStrategy);
    }

    /**
     * Get the number of devices
     * @return the number of devices
     */
    public int deviceNum() {
        return numDevices;
    }

    /**
     * Get the configuration for the current
     * device and thread
     * @return the current configuration for
     * the given device and thread
     */
    public  DeviceConfiguration getConf() {
        return getConf(getDeviceForThread());
    }


    /**
     * Get the memory strategy for the current thread
     * and device
     * @return
     */
    public MemoryStrategy getMemoryStrategy() {
        // FIXME: this ad-hoc is used to get forced strategies working for initial pass on CUDA mem allocation tests, and this could/should be removed before release
        if (forcedStrategies.containsKey(Thread.currentThread().getName())) return forcedStrategies.get(Thread.currentThread().getName());
            else return getConf().getMemoryStrategy();
    }




    /**
     * Configure the given information
     * based on the device
     */
    public void configure() {
        if(confCalled)
            return;

        JCublas2.setExceptionsEnabled(true);
        JCudaDriver.setExceptionsEnabled(true);
        JCuda.setExceptionsEnabled(true);

        if(deviceSetup.get())
            return;

        JCudaDriver.cuInit(0);
        int count[] = new int[1];
        cuDeviceGetCount(count);
        numDevices = count[0];
        log.debug("Found " + numDevices + " gpus");

        if(numDevices < 1)
            numDevices = 1;
        bannedDevices = new ArrayList<>();


        String props = System.getProperty(DEVICES_TO_BAN, "-1");
        String[] split = props.split(",");
        //Should only be used in multi device scenarios; otherwise always use one device
        if(split.length >= 1)
            for(String s : split) {
                Integer i = Integer.parseInt(s);
                if(i >= 0)
                    bannedDevices.add(Integer.parseInt(s));

            }

        deviceSetup.set(true);
        Set<Thread> threadSet = Thread.getAllStackTraces().keySet();

        for(int i = 0; i < numDevices; i++) {
            for(Thread thread : threadSet)
                getContext(i,thread.getName());
        }


        setContext();

      

        // Check if the device supports mapped host memory
        cudaDeviceProp deviceProperties = new cudaDeviceProp();
        JCuda.cudaGetDeviceProperties(deviceProperties, 0);
        if (deviceProperties.canMapHostMemory == 0) {
            System.err.println("This device can not map host memory");
            System.err.println(deviceProperties.toFormattedString());
            return;
        }


        /*
        // if we'll need stack initialization, here's the code
        int numberOfCores = CudaArgs.convertMPtoCores(deviceProperties.major, deviceProperties.minor, deviceProperties.multiProcessorCount) * deviceProperties.multiProcessorCount;
        int maxThreadsPerCore = deviceProperties.maxThreadsPerMultiProcessor / CudaArgs.convertMPtoCores(deviceProperties.major, deviceProperties.minor, deviceProperties.multiProcessorCount);


        long stackSize = Math.min(512*1024, deviceProperties.totalGlobalMem / numberOfCores  / (maxThreadsPerCore + 8) );

        JCuda.cudaDeviceSetLimit(0,stackSize);

        */



        //force certain ops to have a certain number of threads
        Properties threadProps = new Properties();
        try {
            InputStream is = ContextHolder.class.getResourceAsStream("/function_threads.properties");
            threadProps.load(is);
        } catch (IOException e) {
            e.printStackTrace();
        }

        for(String prop : threadProps.stringPropertyNames()) {
            threads.put(prop,Integer.parseInt(threadProps.getProperty(prop)));
        }

        try {
            GenericObjectPoolConfig config = new GenericObjectPoolConfig();
            config.setJmxEnabled(true);
            config.setBlockWhenExhausted(false);
            config.setMaxIdle(Runtime.getRuntime().availableProcessors());
            config.setMaxTotal(Runtime.getRuntime().availableProcessors());
            config.setMinIdle(Runtime.getRuntime().availableProcessors());
            config.setJmxNameBase("handles");
            handlePool = new CublasHandlePool(new CublasHandlePooledItemFactory(),config);
            GenericObjectPoolConfig confClone = config.clone();
            confClone.setMaxTotal(Runtime.getRuntime().availableProcessors() * 10);
            confClone.setMaxIdle(Runtime.getRuntime().availableProcessors() * 10);
            confClone.setMinIdle(Runtime.getRuntime().availableProcessors() * 10);
            GenericObjectPoolConfig streamConf = confClone.clone();
            streamConf.setJmxNameBase("streams");
            streamPool = new StreamPool(new StreamItemFactory(),streamConf);
            GenericObjectPoolConfig oldStreamConf = streamConf.clone();
            oldStreamConf.setJmxNameBase("oldstream");
            oldStreamPool = new OldStreamPool(new OldStreamItemFactory(),oldStreamConf);
            setContext();
            //seed with multiple streams to encourage parallelism
            for(int i = 0; i < Runtime.getRuntime().availableProcessors(); i++) {
                streamPool.addObject();
                oldStreamPool.addObject();
            }


            //force context initialization to occur
            JCuda.cudaFree(Pointer.to(new int[]{0}));

        }catch(Exception e) {
            log.warn("Unable to initialize cuda",e);
        }


        for(int i = 0; i < numDevices; i++) {
            ClassPathResource confFile = new ClassPathResource("devices/" + i, ContextHolder.class.getClassLoader());
            if(confFile.exists()) {
                Properties props2 = new Properties();
                try {
                    props2.load(confFile.getInputStream());
                    confs.put(i,new DeviceConfiguration(i,props2));
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }

            }
            else
                confs.put(i,new DeviceConfiguration(i));

        }

        confCalled = true;
    }

    public void setNumDevices(int numDevices) {
        this.numDevices = numDevices;
    }


    /**
     * Returns the gpu diagnostics
     * for the current thread
     * @return
     */
    public GpuInformation getCurrentGpuInformation() {
        return getGpuInfo(getDeviceForThread());
    }

    /**
     * Returns the gpu diagnostics
     * for the given device
     * @param device the device to get
     *               the gpu diagnostics for
     * @return
     */
    public GpuInformation getGpuInfo(int device) {
        return info.get(device);
    }


    public Map<Integer, GpuInformation> getInfo() {
        return info;
    }

    /**
     * Get the configuration the given device
     * @param device the device to get the configuration for
     * @return the device configuration
     */
    public DeviceConfiguration getConf(int device) {
        return confs.get(device);
    }


    /**
     * Sets the current context, device
     * and proper device flags
     */
    public synchronized void setContext() {
        JCudaDriver.cuCtxSetCurrent(getContext());
    }

    /**
     * Get the device number for a particular host thread
     * @return the device for the given host thread
     *
     */
    public int getDeviceForThread() {
        if(numDevices > 1) {
            Integer device =  threadNameToDeviceNumber.get(Thread.currentThread().getName());
            if(device == null) {
                org.nd4j.linalg.api.rng.Random random = Nd4j.getRandom();
                if(random == null)
                    throw new IllegalStateException("Unable to load random class");
                device = Nd4j.getRandom().nextInt(numDevices);
                //reroute banned devices
                while(bannedDevices != null && bannedDevices.contains(device))
                    device = Nd4j.getRandom().nextInt(numDevices);
                threadNameToDeviceNumber.put(Thread.currentThread().getName(),device);
                return device;
            }
        }

        return 0;
    }


    /**
     * Get the handle for the current thread
     * @return the handle for the current thread
     */
    public  cublasHandle getHandle() {
        cublasHandle handle =  handleMap.get(Thread.currentThread().getName());
        if(handle != null)
            return handle;
        handle = new cublasHandle();
        JCublas2.cublasCreate(handle);
        handleMap.put(Thread.currentThread().getName(),handle);
        return handle;
    }

    /**
     * Retrieve a context for use with the current thread
     * and the given device
     * @return the context for the given device and thread
     */
    public  CUcontext getContext() {
        return getContext(getDeviceForThread(),Thread.currentThread().getName());
    }

    /**
     * Get the stream for the current thread
     * based on the device for the thread
     * @return the stream for the device and
     * thread
     */
    public synchronized cudaStream_t getCudaStream() {
        Thread currentThread = Thread.currentThread();
        CUcontext ctx = getContext(getDeviceForThread(),currentThread.getName());
        cudaStream_t stream = cudaStreams.get(ctx, currentThread.getName());

        if(stream == null) {
            stream = new cudaStream_t();
            checkResult(JCudaDriver.cuCtxSetCurrent(ctx));
            JCuda.cudaStreamCreate(stream);
            checkResult(JCuda.cudaStreamCreate(stream));
            cudaStreams.put(ctx, currentThread.getName(), stream);
        }

        return stream;
    }



    private void checkResult(int result) {
        if (result != CUresult.CUDA_SUCCESS) {
            throw new CudaException("Failed to create a stream: "+ CUresult.stringFor(result));
        }
    }

    /**
     * Retrieve a context for use with the current thread
     * and the given device
     * @param deviceToUse the device to use
     * @return the t
     */
    public  synchronized CUcontext getContext(int deviceToUse,String threadName) {
        CUcontext ctx = deviceIDContexts.get(deviceToUse,threadName);
        if(ctx == null) {
            ctx = new CUcontext();
            for(int device = 0; device < numDevices; device++) {
                initialize(ctx,device);
                CUdevice currDevice = createDevice(ctx, device);
                devices.put(device,currDevice);
                info.put(device, new GpuInformation(currDevice));
                deviceIDContexts.put(device,threadName,ctx);
            }

        }

        return ctx;
    }


    /**
     * Initializes this KernelLauncher. This method will try to
     * initialize the JCuda driver API. Then it will try to
     * attach to the current CUDA context. If no active CUDA
     * context exists, then it will try to create one, for
     * the device which is specified by the current
     * deviceNumber.
     *
     * @throws CudaException If it is neither possible to
     * attach to an existing context, nor to create a new
     * context.
     */
    private void initialize(CUcontext context,int deviceNumber) {
        // Try to obtain the current context
        cuCtxGetCurrent(context);


        // If the context is 'null', then a new context
        // has to be created.
        CUcontext nullContext = new CUcontext();
        if (context.equals(nullContext))
            createContext(context,deviceNumber);

    }

    /**
     * Tries to create a context for device 'deviceNumber'.
     *
     * @throws CudaException If the device can not be
     * accessed or the context can not be created
     */
    private void createContext(CUcontext context,int deviceNumber) {
        CUdevice device = new CUdevice();
        int result = cuDeviceGet(device, deviceNumber);
        if (result != CUresult.CUDA_SUCCESS) {
            throw new CudaException(
                    "Failed to obtain a device: "+
                            CUresult.stringFor(result));
        }

        result = cuCtxCreate(context, 0, device);
        if (result != CUresult.CUDA_SUCCESS) {
            throw new CudaException(
                    "Failed to create a context: "+
                            CUresult.stringFor(result));
        }

    }

    /**
     * Create a context for the given device
     * @param context the context to create
     * @param deviceNumber the device number to create the context for
     * @return the created device
     */
    public static CUdevice createDevice(CUcontext context,int deviceNumber) {
        CUdevice device = new CUdevice();
        int result = cuDeviceGet(device, deviceNumber);
        if (result != CUresult.CUDA_SUCCESS) {
            throw new CudaException(
                    "Failed to obtain a device: "+
                            CUresult.stringFor(result));
        }

        result = cuCtxCreate(context, 0, device);
        if (result != CUresult.CUDA_SUCCESS) {
            throw new CudaException(
                    "Failed to create a context: "+
                            CUresult.stringFor(result));
        }

        return device;
    }

    /**
     * Get the information for a particular device
     * @param cUdevice the device to get the info for
     * @return the information for a particular device
     */
    public  GpuInformation getInfoFor(int cUdevice) {
        getContext(cUdevice,Thread.currentThread().getName());
        return info.get(cUdevice);
    }

    /**
     * Shutdown this instance
     */
    public synchronized  void destroy() {
        if(shutdown.get())
            return;
        shutdown.set(true);
    }

}
