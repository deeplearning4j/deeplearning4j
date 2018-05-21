package org.nd4j.linalg.primitives;

import lombok.*;

import java.io.Serializable;

/**
 * Simple triple elements holder implementation
 * @author raver119@gmail.com
 */
@Data
@AllArgsConstructor
@Builder
public class ImmutableTriple<F, S, T> implements Serializable {
    private static final long serialVersionUID = 119L;

    protected ImmutableTriple() {

    }

    @Setter(AccessLevel.NONE) protected F first;
    @Setter(AccessLevel.NONE) protected S second;
    @Setter(AccessLevel.NONE) protected T third;


    public F getLeft() {
        return first;
    }

    public S getMiddle() {
        return second;
    }

    public T getRight() {
        return third;
    }

    public static <F, S, T> ImmutableTriple<F, S,T> tripleOf(F first, S second, T third) {
        return new ImmutableTriple<F, S, T>(first, second, third);
    }

    public static <F, S, T> ImmutableTriple<F, S,T> of(F first, S second, T third) {
        return new ImmutableTriple<F, S, T>(first, second, third);
    }
}
