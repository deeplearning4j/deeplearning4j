package org.nd4j.etl4j.api.transform.comparator;

import org.nd4j.etl4j.api.writable.Writable;

import java.io.Serializable;
import java.util.Comparator;

public class FloatWritableComparator implements Comparator<Writable>, Serializable {
    @Override
    public int compare(Writable o1, Writable o2) {
        return Float.compare(o1.toFloat(), o2.toFloat());
    }
}
