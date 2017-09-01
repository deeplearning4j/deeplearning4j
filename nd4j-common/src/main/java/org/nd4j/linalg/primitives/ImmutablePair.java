package org.nd4j.linalg.primitives;

import lombok.*;

import java.io.Serializable;

/**
 * Simple pair implementation
 *
 * @author raver119@gmail.com
 */
@AllArgsConstructor
@Data
@Builder
public class ImmutablePair<K, V> implements Serializable {
    private static final long serialVersionUID = 119L;

    protected ImmutablePair() {
        //
    }

    @Setter(AccessLevel.NONE) protected K key;
    @Setter(AccessLevel.NONE) protected V value;

    public K getLeft() {
        return key;
    }

    public V getRight() {
        return value;
    }

    public K getFirst() {
        return key;
    }

    public V getSecond() {
        return value;
    }


    public static <T, E> ImmutablePair<T,E> of(T key, E value) {
        return new ImmutablePair<T, E>(key, value);
    }

    public static <T, E> ImmutablePair<T,E> makePair(T key, E value) {
        return new ImmutablePair<T, E>(key, value);
    }

    public static <T, E> ImmutablePair<T,E> create(T key, E value) {
        return new ImmutablePair<T, E>(key, value);
    }

    public static <T, E> ImmutablePair<T,E> pairOf(T key, E value) {
        return new ImmutablePair<T, E>(key, value);
    }
}
