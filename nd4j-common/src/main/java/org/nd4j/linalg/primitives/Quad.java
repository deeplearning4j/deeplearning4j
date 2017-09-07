package org.nd4j.linalg.primitives;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;

/**
 * Simple triple elements holder implementation
 * @author raver119@gmail.com
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Quad<F, S, T, O> implements Serializable {
    private static final long serialVersionUID = 119L;

    protected F first;
    protected S second;
    protected T third;
    protected O fourth;

    public static <F, S, T, O> Quad<F, S, T, O> quadOf(F first, S second, T third, O fourth) {
        return new Quad<>(first, second, third, fourth);
    }

    public static <F, S, T, O> Quad<F, S,T, O> of(F first, S second, T third, O fourth) {
        return new Quad<>(first, second, third, fourth);
    }
}
