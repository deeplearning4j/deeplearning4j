package org.nd4j.linalg.primitives;

import com.google.common.util.concurrent.AtomicDouble;

import java.io.Serializable;
import java.util.Collection;
import java.util.HashMap;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Simple counter implementation
 * @author
 */
public class Counter<T> implements Serializable {
    private static final long serialVersionUID = 119L;

    protected HashMap<T, AtomicDouble> map;
    protected AtomicDouble totalCount = new AtomicDouble(0);


    public double getCount(T element) {
        return map.get(element).get();
    }

    public void incrementCount(T element, double inc) {
        AtomicDouble t = map.get(element);
        if (t != null)
            t.addAndGet(inc);
        else {
            map.put(element, new AtomicDouble(inc));
        }

        totalCount.addAndGet(inc);
    }

    /**
     * This method will increment all elements in collection
     *
     * @param elements
     * @param inc
     */
    public void incrementAll(Collection<T> elements, double inc) {
        for (T element: elements) {
            incrementCount(element, inc);
        }
    }

    /**
     * This method returns probability of given element
     *
     * @param element
     * @return
     */
    public double getProbability(T element) {
        if (totalCount.get() <= 0.0)
            throw new IllegalStateException("Can't calculate probability with empty counter");

        return getCount(element) / totalCount.get();
    }

    /**
     * This method sets new counter value for given element
     *
     * @param element element to be updated
     * @param count new counter value
     * @return previous value
     */
    public double setCount(T element, double count) {
        AtomicDouble t = map.get(element);
        if (t != null)
            return t.getAndSet(count);
        else {
            map.put(element, new AtomicDouble(count));
            return 0;
        }

    }

    /**
     * This method returns Set of elements used in this counter
     *
     * @return
     */
    public Set<T> keySet() {
        return map.keySet();
    }

    public void normalize() {
        for (T key : keySet()) {
            setCount(key, getCount(key) / totalCount.get());
        }

        rebuildTotals();
    }

    protected void rebuildTotals() {
        totalCount.set(0);
        for (T key : keySet()) {
            totalCount.addAndGet(getCount(key));
        }
    }
}
