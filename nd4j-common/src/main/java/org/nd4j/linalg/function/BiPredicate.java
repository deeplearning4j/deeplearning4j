package org.nd4j.linalg.function;

/**
 * A predicate (boolean valued function) with two arguments.
 *
 * @param <T> Type of first argument
 * @param <U> Type of second argument
 */
public interface BiPredicate<T, U> {

    /**
     * Evaluate the predicate
     *
     * @param t First argument
     * @param u Second argument
     * @return Result
     */
    boolean test(T t, U u);

}
