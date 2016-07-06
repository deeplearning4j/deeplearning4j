package org.nd4j.etl4j.api.transform.comparator;

import org.canova.api.writable.Writable;

import java.io.Serializable;
import java.util.Comparator;

public class DoubleWritableComparator implements Comparator<Writable>, Serializable {
    @Override
    public int compare(Writable o1, Writable o2) {
        return Double.compare(o1.toDouble(), o2.toDouble());
    }
}
