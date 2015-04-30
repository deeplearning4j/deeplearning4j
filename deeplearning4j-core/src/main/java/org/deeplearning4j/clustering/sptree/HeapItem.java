package org.deeplearning4j.clustering.sptree;

import java.io.Serializable;

/**
 * @author Adam Gibson
 */
public class HeapItem implements Serializable,Comparable<HeapItem> {
    private int index;
    private double distance;


    public HeapItem(int index, double distance) {
        this.index = index;
        this.distance = distance;
    }

    public int getIndex() {
        return index;
    }

    public void setIndex(int index) {
        this.index = index;
    }

    public double getDistance() {
        return distance;
    }

    public void setDistance(double distance) {
        this.distance = distance;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        HeapItem heapItem = (HeapItem) o;

        if (index != heapItem.index) return false;
        return Double.compare(heapItem.distance, distance) == 0;

    }

    @Override
    public int hashCode() {
        int result;
        long temp;
        result = index;
        temp = Double.doubleToLongBits(distance);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        return result;
    }

    @Override
    public int compareTo(HeapItem o) {
        return distance < o.distance ? 1 : 0;
    }
}
