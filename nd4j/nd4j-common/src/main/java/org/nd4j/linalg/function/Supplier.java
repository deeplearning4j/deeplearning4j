package org.nd4j.linalg.function;

/**
 * A supplier of results with no input arguments
 *
 * @param <T> Type of result
 */
public interface Supplier<T> {

    /**
     * @return Result
     */
    T get();

}
