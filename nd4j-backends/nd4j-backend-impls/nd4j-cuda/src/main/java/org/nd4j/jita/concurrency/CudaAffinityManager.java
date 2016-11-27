package org.nd4j.jita.concurrency;

import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.concurrency.BasicAffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author raver119@gmail.com
 */
public class CudaAffinityManager extends BasicAffinityManager {

    private static final Configuration configuration = CudaEnvironment.getInstance().getConfiguration();

    private static Logger logger = LoggerFactory.getLogger(CudaAffinityManager.class);

    private Map<Long, Integer> affinityMap = new ConcurrentHashMap<>();
    private AtomicInteger devPtr = new AtomicInteger(0);
    private ThreadLocal<AtomicBoolean> affiliated = new ThreadLocal<>();

    private AtomicInteger numberOfDevices = new AtomicInteger(-1);

    public CudaAffinityManager() {
        super();
    }

    @Override
    public Integer getDeviceForCurrentThread() {
        return getDeviceForThread(Thread.currentThread().getId());
    }

    @Override
    public Integer getDeviceForThread(Thread thread) {
        return getDeviceForThread(thread.getId());
    }

    @Override
    public Integer getDeviceForThread(long threadId) {
        if (!affinityMap.containsKey(threadId)) {
            Integer deviceId = getNextDevice(threadId);
            affinityMap.put(threadId, deviceId);
            affiliated.set(new AtomicBoolean(false));

            if (threadId == Thread.currentThread().getId()) {
                NativeOpsHolder.getInstance().getDeviceNativeOps().setDevice(new CudaPointer(deviceId));
         //       logger.debug("setDevice({}) called for thread {}", deviceId, threadId);
                affiliated.get().set(true);
            }

            return deviceId;
        }

        if (threadId == Thread.currentThread().getId()) {
            if (affiliated.get() == null)
                affiliated.set(new AtomicBoolean(false));

            if (!affiliated.get().get()) {
                int deviceId = affinityMap.get(threadId);
                NativeOpsHolder.getInstance().getDeviceNativeOps().setDevice(new CudaPointer(deviceId));
        //        logger.debug("SCARY setDevice({}) called for thread {}", deviceId, threadId);
                affiliated.get().set(true);
                return deviceId;
            }
        }

        return affinityMap.get(threadId);
    }

    @Override
    public void attachThreadToDevice(Thread thread, Integer deviceId) {
        attachThreadToDevice(thread.getId(), deviceId);
    }

    @Override
    public void attachThreadToDevice(long threadId, Integer deviceId) {
        List<Integer> devices = new ArrayList<>(configuration.getAvailableDevices());
        logger.debug("Manually mapping thread [{}] to device [{}], out of [{}] devices...", threadId , deviceId, devices.size());
        affinityMap.put(threadId, deviceId);
    }

    protected Integer getNextDevice(long threadId) {
        List<Integer> devices = new ArrayList<>(configuration.getAvailableDevices());
        Integer device = null;
        if (!configuration.isForcedSingleGPU()) {
            // simple round-robin here
            synchronized (this) {
                device = devices.get(devPtr.getAndIncrement());

                // We check only for number of entries here, not their actual values
                if (devPtr.get() >= devices.size())
                    devPtr.set(0);

                logger.debug("Mapping thread [{}] to device [{}], out of [{}] devices...", threadId , device, devices.size());
            }
        } else {
            device = configuration.getAvailableDevices().get(0);
            logger.debug("Single device is forced, mapping to device [{}]", device);
        }

        return device;
    }

    @Override
    public int getNumberOfDevices() {
        if (numberOfDevices.get() < 0) {
            synchronized (this) {
                if (numberOfDevices.get() < 1) {
                    numberOfDevices.set(NativeOpsHolder.getInstance().getDeviceNativeOps().getAvailableDevices());
                }
            }
        }

        return numberOfDevices.get();
    }

    /**
     * Utility method, to associate INDArray with specific device (backend-specific)
     *
     * @param array
     */
    @Override
    public void touch(INDArray array) {
        if (array == null)
            return;

        touch(array.data());
        touch(array.shapeInfoDataBuffer());
    }

    /**
     * Utility method, to associate INDArray with specific device (backend-specific)
     *
     * @param buffer
     */
    @Override
    public void touch(DataBuffer buffer) {
        if (buffer == null)
            return;

        AllocationPoint point = AtomicAllocator.getInstance().getAllocationPoint(buffer);

        if (point.isConstant()) {
            Nd4j.getConstantHandler().relocateConstantSpace(buffer);
        } else {
            AtomicAllocator.getInstance().getMemoryHandler().relocateObject(buffer);
        }
    }

    /**
     * This method replicates given INDArray, and places it to target device.
     *
     * @param deviceId target deviceId
     * @param array    INDArray to replicate
     * @return
     */
    @Override
    public synchronized INDArray replicateToDevice(Integer deviceId, INDArray array) {
        if (array == null)
            return null;

        if (array.isView())
            throw new UnsupportedOperationException("It's impossible to replicate View");

        int[] shape = array.shape();
        int[] stride = array.stride();
        int elementWiseStride = array.elementWiseStride();
        char ordering = array.ordering();
        int length = array.length();

        // we use this call to get device memory updated
        AtomicAllocator.getInstance().getPointer(array, (CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext());

        int currentDeviceId = getDeviceForCurrentThread();

        NativeOpsHolder.getInstance().getDeviceNativeOps().setDevice(new CudaPointer(deviceId));
        attachThreadToDevice(Thread.currentThread().getId(), deviceId);


        DataBuffer newDataBuffer =  replicateToDevice(deviceId, array.data());
        DataBuffer newShapeBuffer = Nd4j.getShapeInfoProvider().createShapeInformation(shape, stride, 0, elementWiseStride, ordering);
        INDArray result = Nd4j.createArrayFromShapeBuffer(newDataBuffer, newShapeBuffer);

        attachThreadToDevice(Thread.currentThread().getId(), currentDeviceId);
        NativeOpsHolder.getInstance().getDeviceNativeOps().setDevice(new CudaPointer(currentDeviceId));


        return result;
    }

    /**
     * This method replicates given DataBuffer, and places it to target device.
     *
     * @param deviceId target deviceId
     * @param buffer
     * @return
     */
    @Override
    public DataBuffer replicateToDevice(Integer deviceId, DataBuffer buffer) {
        if (buffer == null)
            return null;

        int currentDeviceId = AtomicAllocator.getInstance().getDeviceId();
        if (currentDeviceId != deviceId) {
            NativeOpsHolder.getInstance().getDeviceNativeOps().setDevice(new CudaPointer(deviceId));
            Nd4j.getAffinityManager().attachThreadToDevice(Thread.currentThread().getId(), deviceId);
        }

        DataBuffer dstBuffer = Nd4j.createBuffer(buffer.length(), false);
        AtomicAllocator.getInstance().memcpy(dstBuffer, buffer);

        if (currentDeviceId != deviceId) {
            NativeOpsHolder.getInstance().getDeviceNativeOps().setDevice(new CudaPointer(currentDeviceId));
            Nd4j.getAffinityManager().attachThreadToDevice(Thread.currentThread().getId(), currentDeviceId);
        }

        return dstBuffer;
    }

    @Override
    public void tagLocation(INDArray array, Location location) {
        if (location == Location.HOST)
            AtomicAllocator.getInstance().getAllocationPoint(array).tickHostWrite();
        else if (location == Location.DEVICE)
            AtomicAllocator.getInstance().getAllocationPoint(array).tickDeviceWrite();
    }

    @Override
    public void tagLocation(DataBuffer buffer, Location location) {
        if (location == Location.HOST)
            AtomicAllocator.getInstance().getAllocationPoint(buffer).tickHostWrite();
        else if (location == Location.DEVICE)
            AtomicAllocator.getInstance().getAllocationPoint(buffer).tickDeviceWrite();
    }
}
