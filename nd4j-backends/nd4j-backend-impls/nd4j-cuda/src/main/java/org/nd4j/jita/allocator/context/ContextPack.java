package org.nd4j.jita.allocator.context;

import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import org.apache.commons.lang3.RandomUtils;
import org.nd4j.linalg.jcublas.context.CudaContext;

import java.util.HashMap;
import java.util.Map;

/**
 * @author raver119@gmail.com
 */
public class ContextPack {
    @Getter
    @Setter
    private Integer deviceId;
    @Getter
    private int availableLanes;
    private Map<Integer, CudaContext> lanes = new HashMap<>();

    public ContextPack(int totalLanes) {
        availableLanes = totalLanes;
    }

    public ContextPack(CudaContext context) {
        this.availableLanes = 1;
        lanes.put(0, context);
    }

    public void addLane(@NonNull Integer laneId, @NonNull CudaContext context) {
        lanes.put(laneId, context);
        context.setLaneId(laneId);
    }

    public CudaContext getContextForLane(Integer laneId) {
        return lanes.get(laneId);
    }

    public int nextRandomLane() {
        if (availableLanes == 1)
            return 0;
        return RandomUtils.nextInt(0, availableLanes);
    }
}
