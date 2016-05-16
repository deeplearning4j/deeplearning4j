package org.nd4j.jita.allocator.tad;

import org.apache.commons.math3.util.Pair;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.jcublas.CachedShapeInfoProvider;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Semaphore;

/**
 * @author raver119@gmail.com
 */
public class DeviceTADManager extends BasicTADManager {
    protected List<Map<TadDescriptor, Pair<DataBuffer, DataBuffer>>> tadCache = new ArrayList<>();
    private Semaphore lock = new Semaphore(1);
    private static Logger logger = LoggerFactory.getLogger(DeviceTADManager.class);
    private Configuration configuration = CudaEnvironment.getInstance().getConfiguration();

    public DeviceTADManager() {
        int numDevices =  configuration.getAvailableDevices().size();

        for (int i = 0; i < numDevices; i++ ) {
            tadCache.add(i, new ConcurrentHashMap<TadDescriptor, Pair<DataBuffer, DataBuffer>>());
        }
    }

    @Override
    public Pair<DataBuffer, DataBuffer> getTADOnlyShapeInfo(INDArray array, int[] dimension) {
        /*
            so, we check, if we have things cached. If we don't - we just create new TAD shape, and push it to constant memory
        */

        Integer deviceId = AtomicAllocator.getInstance().getDeviceId();

      //  logger.info("Requested TAD for device [{}], dimensions: [{}]", deviceId, Arrays.toString(dimension));

        TadDescriptor descriptor = new TadDescriptor(array, dimension);


        if (!tadCache.get(deviceId).containsKey(descriptor)) {
       //     logger.info("Creating new TAD...");
            Pair<DataBuffer, DataBuffer>buffers = super.getTADOnlyShapeInfo(array, dimension);

            if (buffers.getFirst() != array.shapeInfoDataBuffer())
                AtomicAllocator.getInstance().moveToConstant(buffers.getFirst());

            if (buffers.getSecond() != null)
                AtomicAllocator.getInstance().moveToConstant(buffers.getSecond());

            // so, at this point we have buffer valid on host side. And we just need to replace DevicePointer with constant pointer
            tadCache.get(deviceId).put(descriptor, buffers);
        } //else logger.info("Using TAD from cache...");

        return tadCache.get(deviceId).get(descriptor);
    }
}
