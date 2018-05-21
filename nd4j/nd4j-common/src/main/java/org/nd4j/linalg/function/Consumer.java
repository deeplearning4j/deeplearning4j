package org.nd4j.linalg.function;

/**
 * A function that accepts a single input and returns no result
 *
 * @param <T> Type of the input
 */
public interface Consumer<T> {

    /**
     * Perform the operation on the input
     * @param t Input
     */
    void accept(T t);
}
