package org.nd4j.autodiff;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;

/**
 * Holds a set of arrays keyed by a String name, functioning essentially like a {@code Map<String,INDArray>}.<br>
 * Implementations may have different internal ways of storing arrays, however.<br>
 * For example for single threaded applications: {@link org.nd4j.autodiff.samediff.array.SingleThreadArrayHolder}<br>
 * And for multi-threaded: {@link org.nd4j.autodiff.samediff.array.ThreadSafeArrayHolder}
 *
 * @author Alex Black
 */
public interface ArrayHolder {

    /**
     * @return True if an array by that name exists
     */
    boolean hasArray(String name);

    INDArray getArray(String name);

    void setArray(String name, INDArray array);

    INDArray removeArray(String name);

    int size();

    void initFrom(ArrayHolder arrayHolder);

    Collection<String> arrayNames();

    void rename(String from, String to);
}
