package org.nd4j.linalg.function;

/**
 * A specialization of {@link Function} where the input and return types are the same
 *
 * @param <T> Input and return types
 */
public interface UnaryOperator<T> extends Function<T,T> {
}
