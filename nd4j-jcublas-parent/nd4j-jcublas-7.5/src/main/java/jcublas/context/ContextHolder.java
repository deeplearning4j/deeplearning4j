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

package jcublas.context;

import jcuda.CudaException;
import jcuda.driver.*;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaStream_t;
import org.nd4j.linalg.api.buffer.allocation.MemoryStrategy;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.device.conf.DeviceConfiguration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Properties;
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
public class ContextHolder {

    private Map<Integer,CUdevice> devices = new ConcurrentHashMap<>();
    private Map<Integer,GpuInformation> info = new ConcurrentHashMap<>();
    private Map<Integer, CUcontext> deviceIDContexts = new ConcurrentHashMap<>();
    private Map<String,Integer> threadNameToDeviceNumber = new ConcurrentHashMap<>();
    private Table<CUcontext,String,CUstream> contextStreams = HashBasedTable.create();
    private Table<CUcontext,String,cudaStream_t> cudaStreams = HashBasedTable.create();
    private Map<String, cublasHandle> handleMap = new ConcurrentHashMap<>();
    private List<Integer> bannedDevices;
    private int numDevices = 0;
    private Map<Integer,DeviceConfiguration> confs = new ConcurrentHashMap<>();
    private static ContextHolder INSTANCE;
    public final static String DEVICES_TO_BAN = "org.nd4j.linalg.jcuda.jcublas.ban_devices";
    public final static String SYNC_THREADS = "org.nd4j.linalg.jcuda.jcublas.syncthreads";
    private static boolean syncThreads = true;
    private boolean confCalled = false;
    private static Logger log = LoggerFactory.getLogger(ContextHolder.class);
    private AtomicBoolean shutdown = new AtomicBoolean(false);

    private ContextHolder(){
        try {
            getNumDevices();
        }catch(Exception e) {
            log.warn("Unable to initialize cuda",e);
        }
    }

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
    public  MemoryStrategy getMemoryStrategy() {
        return getConf().getMemoryStrategy();
    }


    /**
     * Configure the given information
     * based on the device
     */
    public void configure() {
        if(confCalled)
            return;


        syncThreads = Boolean.parseBoolean(System.getProperty(SYNC_THREADS,"true"));
        if(numDevices == 0) {
            getNumDevices();
        }

        for(int i = 0; i < numDevices; i++) {
            ClassPathResource confFile = new ClassPathResource("devices/" + i, ContextHolder.class.getClassLoader());
            if(confFile.exists()) {
                Properties props = new Properties();
                try {
                    props.load(confFile.getInputStream());
                    confs.put(i,new DeviceConfiguration(i,props));
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
     * Get the configuration the given device
     * @param device the device to get the configuration for
     * @return the device configuration
     */
    public DeviceConfiguration getConf(int device) {
        return confs.get(device);
    }

    private void getNumDevices() {
        JCudaDriver.setExceptionsEnabled(true);
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


    }


    /**
     * Synchronized the stream.
     * This should be run after
     * every operation.
     */
    public static void syncStream() {
        JCudaDriver.cuCtxSetCurrent(getInstance().getContext());
        //old api
        JCublas2.cublasSetStream(getInstance().getHandle(), getInstance().getCudaStream());
        JCuda.cudaStreamSynchronize(getInstance().getCudaStream());
        //new api
        JCudaDriver.cuStreamSynchronize(getInstance().getStream());
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
        return getContext(getDeviceForThread());
    }

    /**
     * Get the stream for the current thread
     * based on the device for the thread
     * @return the stream for the device and
     * thread
     */
    public synchronized cudaStream_t getCudaStream() {
        Thread currentThread = Thread.currentThread();
        CUcontext ctx = getContext(getDeviceForThread());
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


    /**
     * Get the stream for the current thread
     * based on the device for the thread
     * @return the stream for the device and
     * thread
     */
    public  CUstream getStream() {
        Thread currentThread = Thread.currentThread();
        CUcontext ctx = getContext(getDeviceForThread());
        CUstream stream = contextStreams.get(ctx, currentThread.getName());

        if(stream == null) {
            stream = new CUstream();
            checkResult(JCudaDriver.cuCtxSetCurrent(ctx));
            checkResult(JCudaDriver.cuStreamCreate(stream, CUstream_flags.CU_STREAM_NON_BLOCKING));
            contextStreams.put(ctx, currentThread.getName(), stream);
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
    public  synchronized CUcontext getContext(int deviceToUse) {

        CUcontext ctx = deviceIDContexts.get(deviceToUse);
        if(ctx == null) {
            ctx = new CUcontext();
            for(int device = 0; device < numDevices; device++) {
                initialize(ctx,device);
                CUdevice currDevice = createDevice(ctx, device);
                devices.put(device,currDevice);
                info.put(device, new GpuInformation(currDevice));
                deviceIDContexts.put(device,ctx);
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
        cuInit(0);

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
        getContext(cUdevice);
        return info.get(cUdevice);
    }

    /**
     * Returns the available devices
     * delimited by device,thread
     * @return the available devices
     */
    public Map<Integer, CUdevice> getDevices() {
        return devices;
    }

    /**
     * Returns the available contexts
     * based on device and thread name
     * @return the context
     */
    public Map<Integer, CUcontext> getDeviceIDContexts() {
        return deviceIDContexts;
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
