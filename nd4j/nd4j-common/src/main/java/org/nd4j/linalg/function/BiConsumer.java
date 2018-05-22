package org.nd4j.linalg.function;

/**
 * BiConsumer is an operation that accepts two arguments and returns no result.
 *
 * @param <T> Type of first argument
 * @param <U> Type of second argument
 */
public interface BiConsumer<T, U> {

    /**
     * Perform the operation on the given arguments
     *
     * @param t First input
     * @param u Second input
     */
    void accept(T t, U u);

}
