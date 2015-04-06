package org.deeplearning4j.clustering.vptree;

import java.io.Serializable;

/**
 * @author Adam Gibson
 */
public class HeapItem implements Serializable,Comparable<HeapItem> {
    private int index;
    private double distance;

    @Override
    public int compareTo(HeapItem o) {
        return distance < o.distance ? 1 : 0;
    }
}
