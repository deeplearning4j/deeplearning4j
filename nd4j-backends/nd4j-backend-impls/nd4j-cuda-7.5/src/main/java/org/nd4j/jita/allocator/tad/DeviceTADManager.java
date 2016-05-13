package org.nd4j.jita.allocator.tad;

import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.jcublas.CachedShapeInfoProvider;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Semaphore;

/**
 * @author raver119@gmail.com
 */
public class DeviceTADManager extends BasicTADManager {
    protected Map<Integer, Map<TadDescriptor, DataBuffer>> tadCache = new ConcurrentHashMap<>();
    private Semaphore lock = new Semaphore(1);
    private static Logger logger = LoggerFactory.getLogger(DeviceTADManager.class);

    @Override
    public DataBuffer getTADOnlyShapeInfo(INDArray array, int[] dimension) {
        /*
            so, we check, if we have things cached. If we don't - we just create new TAD shape, and push it to constant memory
        */

        Integer deviceId = AtomicAllocator.getInstance().getDeviceId();

      //  logger.info("Requested TAD for device [{}], dimensions: [{}]", deviceId, Arrays.toString(dimension));

        TadDescriptor descriptor = new TadDescriptor(array, dimension);
        if (!tadCache.containsKey(deviceId)) {
            try {
                lock.acquire();

                if (!tadCache.containsKey(deviceId))
                    tadCache.put(deviceId, new ConcurrentHashMap<TadDescriptor, DataBuffer>());
            } catch (Exception e) {
                throw new RuntimeException(e);
            } finally {
                lock.release();
            }
        }

        if (!tadCache.get(deviceId).containsKey(descriptor)) {
            //logger.info("Creating new TAD...");
            DataBuffer buffer = super.getTADOnlyShapeInfo(array, dimension);

            AtomicAllocator.getInstance().moveToConstant(buffer);

            // so, at this point we have buffer valid on host side. And we just need to replace DevicePointer with constant pointer
            tadCache.get(deviceId).put(descriptor, buffer);
        } //else logger.info("Using TAD from cache...");

        return tadCache.get(deviceId).get(descriptor);
    }
}
