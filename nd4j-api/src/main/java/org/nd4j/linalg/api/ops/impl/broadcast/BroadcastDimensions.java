package org.nd4j.linalg.api.ops.impl.broadcast;

import com.google.common.primitives.Ints;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by agibsonccc on 11/12/15.
 */
public class BroadcastDimensions {
    /**
     * Determine the valid broad cast dimensions
     * based on which values in the
     * given shape are equal to 1
     * @param shape the shape to get the broadcast
     *              dimensions for
     * @return the dimension indexes with a broadcast shape
     */
    public static int[] getDimensions(int[] shape) {
        List<Integer> getDimensions = new ArrayList<>();
        for(int i = 0; i < shape.length; i++)
            if(shape[i] != 1)
                getDimensions.add(i);
        return Ints.toArray(getDimensions);
    }

}
