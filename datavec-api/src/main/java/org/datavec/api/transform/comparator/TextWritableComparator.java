package org.datavec.api.transform.comparator;

import org.datavec.api.writable.Writable;

import java.io.Serializable;
import java.util.Comparator;

public class TextWritableComparator implements Comparator<Writable>, Serializable {
    @Override
    public int compare(Writable o1, Writable o2) {
        return o1.toString().compareTo(o2.toString());
    }
}
