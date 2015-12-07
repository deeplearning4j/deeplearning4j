package org.deeplearning4j.models.abstractvectors;

import com.google.common.util.concurrent.AtomicDouble;

/**
 * Created by fartovii on 07.12.15.
 */
public abstract class SequenceElement implements Comparable<SequenceElement>{
    protected AtomicDouble elementFrequency = new AtomicDouble(1.0);

    abstract public String getLabel();

    public double getElementFrequency() {
        return elementFrequency.get();
    }

    public void setElementFrequency(double value) {
        this.elementFrequency.set(value);
    }

    @Override
    public int compareTo(SequenceElement o) {
        return Double.compare(elementFrequency.get(), o.elementFrequency.get());
    }
}
