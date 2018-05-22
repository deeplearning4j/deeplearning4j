package org.nd4j.linalg.primitives;

import lombok.*;

import java.io.Serializable;

/**
 * Simple quad elements holder implementation
 * @author raver119@gmail.com
 */
@Data
@AllArgsConstructor
@Builder
public class ImmutableQuad<F, S, T, O> implements Serializable {
    private static final long serialVersionUID = 119L;

    @Setter(AccessLevel.NONE) protected F first;
    @Setter(AccessLevel.NONE) protected S second;
    @Setter(AccessLevel.NONE) protected T third;
    @Setter(AccessLevel.NONE) protected O fourth;

    public static <F, S, T, O> ImmutableQuad<F, S, T, O> quadOf(F first, S second, T third, O fourth) {
        return new ImmutableQuad(first, second, third, fourth);
    }

    public static <F, S, T, O> ImmutableQuad<F, S,T, O> of(F first, S second, T third, O fourth) {
        return new ImmutableQuad(first, second, third, fourth);
    }
}
