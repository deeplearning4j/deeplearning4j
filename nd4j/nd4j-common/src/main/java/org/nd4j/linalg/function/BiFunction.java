package org.nd4j.linalg.function;

/**
 * A function that accepts two arguments and returns a result
 *
 * @param <T> Type of first argument
 * @param <U> Type of second argument
 * @param <R> Type of result
 */
public interface BiFunction<T, U, R> {

    /**
     * Apply the function and return the result
     *
     * @param t First argument
     * @param u Second argument
     * @return Result
     */
    R apply(T t, U u);

}
