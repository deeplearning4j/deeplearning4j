package org.nd4j.jita.constant;

import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.shape.ShapeDescriptor;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * This class implements storage singleton, to guarantee constant buffers persistence
 *
 * @author raver119@gmail.com
 */
public class ConstantProtector {
    private static ConstantProtector ourInstance = new ConstantProtector();

    public static ConstantProtector getInstance() {
        return ourInstance;
    }

    private List<DataBuffer> protectorLegacy = new CopyOnWriteArrayList<>();
    private List<Pair<DataBuffer, int[]>> protector = new CopyOnWriteArrayList<>();
    private List<Map<ShapeDescriptor, Pair<DataBuffer, int[]>>> deviceCache = new ArrayList<>();

    private ConstantProtector() {
        purgeProtector();
    }

    public void purgeProtector() {
        protector = new CopyOnWriteArrayList<>();
        deviceCache = new ArrayList<>();

        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();

        for (int i = 0; i < numDevices; i++) {
            deviceCache.add(i, new ConcurrentHashMap<ShapeDescriptor, Pair<DataBuffer, int[]>>());
        }
    }

    public void persistDataBuffer(DataBuffer buffer) {
        protectorLegacy.add(buffer);
    }

    public void persistDataBuffer(Pair<DataBuffer, int[]> buffer) {
        protector.add(buffer);
    }

    public void persistDataBuffer(int deviceId, ShapeDescriptor descriptor, Pair<DataBuffer, int[]> buffer) {
        deviceCache.get(deviceId).put(descriptor, buffer);
    }

    public Pair<DataBuffer, int[]> getDataBuffer(int deviceId, ShapeDescriptor descriptor) {
        return deviceCache.get(deviceId).get(descriptor);
    }

    public boolean containsDataBuffer(int deviceId, ShapeDescriptor descriptor) {
        return deviceCache.get(deviceId).containsKey(descriptor);
    }


}
