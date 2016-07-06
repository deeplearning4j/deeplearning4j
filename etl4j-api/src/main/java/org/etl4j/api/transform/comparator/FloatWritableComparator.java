package io.skymind.echidna.api.comparator;

import org.canova.api.writable.Writable;

import java.io.Serializable;
import java.util.Comparator;

public class FloatWritableComparator implements Comparator<Writable>, Serializable {
    @Override
    public int compare(Writable o1, Writable o2) {
        return Float.compare(o1.toFloat(), o2.toFloat());
    }
}
