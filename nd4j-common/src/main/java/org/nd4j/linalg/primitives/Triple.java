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
public class Triple<F, S, T> implements Serializable {
    private static final long serialVersionUID = 119L;

    protected F first;
    protected S second;
    protected T third;

    public static <F, S, T> Triple<F, S,T> tripleOf(F first, S second, T third) {
        return new Triple<>(first, second, third);
    }
}
