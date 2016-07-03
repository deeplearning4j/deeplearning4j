package org.nd4j.jita.constant;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.jita.allocator.utils.AllocationUtils;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.jita.flow.FlowController;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.cache.ArrayDescriptor;
import org.nd4j.linalg.cache.ConstantHandler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.CudaDoubleDataBuffer;
import org.nd4j.linalg.jcublas.buffer.CudaFloatDataBuffer;
import org.nd4j.linalg.jcublas.buffer.CudaIntDataBuffer;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Created by raver on 08.06.2016.
 */
public class ProtectedCudaConstantHandler implements ConstantHandler {
    private static ProtectedCudaConstantHandler ourInstance = new ProtectedCudaConstantHandler();

    protected Map<Integer, AtomicLong> constantOffsets = new HashMap<>();
    protected Map<Integer, Semaphore> deviceLocks = new ConcurrentHashMap<>();

    protected Map<Integer, Map<ArrayDescriptor, DataBuffer>> buffersCache = new HashMap<>();
    protected Map<Integer, Pointer> deviceAddresses = new HashMap<>();
    private Configuration configuration = CudaEnvironment.getInstance().getConfiguration();
    protected NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
    protected FlowController flowController;

    protected static final ConstantProtector protector = ConstantProtector.getInstance();

    private static Logger logger = LoggerFactory.getLogger(ProtectedCudaConstantHandler.class);

    private static final int MAX_CONSTANT_LENGTH = 49152;
    private static final int MAX_BUFFER_LENGTH = 272;

    protected Semaphore lock = new Semaphore(1);


    public static ProtectedCudaConstantHandler getInstance() {
        return ourInstance;
    }

    private ProtectedCudaConstantHandler() {
    }

    /**
     * This method moves specified dataBuffer to CUDA constant memory space.
     *
     * PLEASE NOTE: CUDA constant memory is limited to 48KB per device.
     *
     * @param dataBuffer
     * @return
     */
    @Override
    public long moveToConstantSpace(DataBuffer dataBuffer) {
        // now, we move things to constant memory
        Integer deviceId = AtomicAllocator.getInstance().getDeviceId();
        ensureMaps(deviceId);

        AllocationPoint point = AtomicAllocator.getInstance().getAllocationPoint(dataBuffer);

        long requiredMemoryBytes = AllocationUtils.getRequiredMemory(point.getShape());
        // and release device memory :)

        long currentOffset = constantOffsets.get(deviceId).get();
        CudaContext context = (CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext();
        if (currentOffset + requiredMemoryBytes >= MAX_CONSTANT_LENGTH || requiredMemoryBytes > MAX_BUFFER_LENGTH)  {
            if (point.getAllocationStatus() == AllocationStatus.HOST && configuration.getMemoryModel() == Configuration.MemoryModel.DELAYED) {
                AtomicAllocator.getInstance().getMemoryHandler().alloc(AllocationStatus.DEVICE, point, point.getShape(), false);
            }

            nativeOps.memcpyAsync(point.getPointers().getDevicePointer(), point.getPointers().getHostPointer(), requiredMemoryBytes, 1, context.getSpecialStream());
            flowController.commitTransfer(context.getSpecialStream());

            point.setConstant(true);
            point.tickDeviceWrite();
            point.tickHostRead();
            point.setDeviceId(deviceId);

            protector.persistDataBuffer(dataBuffer);

            return 0;
        }

        currentOffset = constantOffsets.get(deviceId).getAndAdd(requiredMemoryBytes);
        if (currentOffset >= MAX_CONSTANT_LENGTH)  {
            if (point.getAllocationStatus() == AllocationStatus.HOST && configuration.getMemoryModel() == Configuration.MemoryModel.DELAYED) {
                AtomicAllocator.getInstance().getMemoryHandler().alloc(AllocationStatus.DEVICE, point, point.getShape(), false);
            }

            nativeOps.memcpyAsync(point.getPointers().getDevicePointer(), point.getPointers().getHostPointer(), requiredMemoryBytes, 1, context.getSpecialStream());
            flowController.commitTransfer(context.getSpecialStream());

            point.setConstant(true);
            point.tickDeviceWrite();
            point.tickHostRead();
            point.setDeviceId(deviceId);

            protector.persistDataBuffer(dataBuffer);

            return 0;
        }

        nativeOps.memcpyConstantAsync(currentOffset, point.getPointers().getHostPointer(), requiredMemoryBytes, 1, context.getSpecialStream());
        flowController.commitTransfer(context.getSpecialStream());

        long cAddr = deviceAddresses.get(deviceId).address() + currentOffset;
        point.setAllocationStatus(AllocationStatus.CONSTANT);
        point.getPointers().setDevicePointer(new CudaPointer(cAddr));
        point.setConstant(true);
        point.tickDeviceWrite();
        point.setDeviceId(deviceId);
        point.tickHostRead();


        protector.persistDataBuffer(dataBuffer);

        return cAddr;
    }

    /**
     * PLEASE NOTE: This method implementation is hardware-dependant.
     * PLEASE NOTE: This method does NOT allow concurrent use of any array
     *
     * @param dataBuffer
     * @return
     */
    @Override
    public DataBuffer relocateConstantSpace(DataBuffer dataBuffer) {
        // we always assume that data is sync, and valid on host side
        Integer deviceId = AtomicAllocator.getInstance().getDeviceId();
        ensureMaps(deviceId);

        if (dataBuffer instanceof CudaIntDataBuffer) {
            int[] data = dataBuffer.asInt();
            return getConstantBuffer(data);
        } else if (dataBuffer instanceof CudaFloatDataBuffer) {
            float[] data = dataBuffer.asFloat();
            return getConstantBuffer(data);
        } else if (dataBuffer instanceof CudaDoubleDataBuffer) {
            double[] data = dataBuffer.asDouble();
            return getConstantBuffer(data);
        }

        throw new IllegalStateException("Unknown CudaDataBuffer type");
    }

    private void ensureMaps(Integer deviceId) {
        if (!buffersCache.containsKey(deviceId)) {
            if (flowController == null)
                flowController = AtomicAllocator.getInstance().getFlowController();

            try {
                lock.acquire();
                if (!buffersCache.containsKey(deviceId)) {
                    buffersCache.put(deviceId, new ConcurrentHashMap<ArrayDescriptor, DataBuffer>());
                    constantOffsets.put(deviceId, new AtomicLong(0));
                    deviceLocks.put(deviceId, new Semaphore(1));

                    Pointer cAddr = nativeOps.getConstantSpace();

                    deviceAddresses.put(deviceId, cAddr);
                }
            } catch (Exception e) {
                throw new RuntimeException(e);
            } finally {
                lock.release();
            }
        }
    }

    /**
     * This method returns DataBuffer with contant equal to input array.
     *
     * PLEASE NOTE: This method assumes that you'll never ever change values within result DataBuffer
     *
     * @param array
     * @return
     */
    @Override
    public DataBuffer getConstantBuffer(int[] array) {
        //  logger.info("getConstantBuffer(int[]) called");
        ArrayDescriptor descriptor = new ArrayDescriptor(array);

        Integer deviceId = AtomicAllocator.getInstance().getDeviceId();

        ensureMaps(deviceId);

        if (!buffersCache.get(deviceId).containsKey(descriptor)) {
            // we create new databuffer
            //    logger.info("Creating new constant buffer...");
            DataBuffer buffer = Nd4j.createBuffer(array);
            buffer.setConstant(true);

            // now we move data to constant memory, and keep happy
            moveToConstantSpace(buffer);

            buffersCache.get(deviceId).put(descriptor, buffer);
            return buffer;
        } //else logger.info("Reusing constant buffer...");

        return buffersCache.get(deviceId).get(descriptor);
    }

    /**
     * This method returns DataBuffer with contant equal to input array.
     *
     * PLEASE NOTE: This method assumes that you'll never ever change values within result DataBuffer
     *
     * @param array
     * @return
     */
    @Override
    public DataBuffer getConstantBuffer(float[] array) {
        //   logger.info("getConstantBuffer(float[]) called");
        ArrayDescriptor descriptor = new ArrayDescriptor(array);

        Integer deviceId = AtomicAllocator.getInstance().getDeviceId();

        ensureMaps(deviceId);

        if (!buffersCache.get(deviceId).containsKey(descriptor)) {
            // we create new databuffer
            //     logger.info("Creating new constant buffer...");
            DataBuffer buffer = Nd4j.createBuffer(array);
            buffer.setConstant(true);

            // now we move data to constant memory, and keep happy
            moveToConstantSpace(buffer);

            buffersCache.get(deviceId).put(descriptor, buffer);
            return buffer;
        } // else logger.info("Reusing constant buffer...");

        return buffersCache.get(deviceId).get(descriptor);
    }

    /**
     * This method returns DataBuffer with contant equal to input array.
     *
     * PLEASE NOTE: This method assumes that you'll never ever change values within result DataBuffer
     *
     * @param array
     * @return
     */
    @Override
    public DataBuffer getConstantBuffer(double[] array) {
//        logger.info("getConstantBuffer(double[]) called");
        ArrayDescriptor descriptor = new ArrayDescriptor(array);

        Integer deviceId = AtomicAllocator.getInstance().getDeviceId();

        ensureMaps(deviceId);

        if (!buffersCache.get(deviceId).containsKey(descriptor)) {
            // we create new databuffer
            //logger.info("Creating new constant buffer...");
            DataBuffer buffer = Nd4j.createBuffer(array);
            buffer.setConstant(true);

            // now we move data to constant memory, and keep happy
            moveToConstantSpace(buffer);

            buffersCache.get(deviceId).put(descriptor, buffer);
            return buffer;
        } //else logger.info("Reusing constant buffer...");

        return buffersCache.get(deviceId).get(descriptor);
    }
}
