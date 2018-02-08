package org.nd4j.linalg.function;

/**
 * A boolean valued function of a single input argument
 *
 * @param <T> Type of the input
 */
public interface Predicate<T> {

    /**
     * Returns the result of the predicate on the given input
     *
     * @param t Input
     * @return Result
     */
    boolean test(T t);

}
