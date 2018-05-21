package org.nd4j.linalg.function;

/**
 * A Function accepts one argument and returns a result
 * @param <T> Type of the argument
 * @param <R> Type of the result
 */
public interface Function<T, R> {

    /**
     * Apply the function to the argument, and return the result
     *
     * @param t Input
     * @return Result
     */
    R apply (T t);

}
