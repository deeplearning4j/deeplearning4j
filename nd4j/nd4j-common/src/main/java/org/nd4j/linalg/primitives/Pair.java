package org.nd4j.linalg.primitives;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Simple pair implementation
 *
 * @author raver119@gmail.com
 */
@AllArgsConstructor
@Data
@NoArgsConstructor
@Builder
public class Pair<K, V> implements Serializable {
    private static final long serialVersionUID = 119L;

    protected K key;
    protected V value;

    @Override
    public String toString() {
        return "Pair{" +
                "key=" + (key instanceof int[] ? Arrays.toString((int[]) key) : key) +
                ", value=" + (value instanceof int[] ? Arrays.toString((int[]) value) : value) +
                '}';
    }

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

    public void setFirst(K first) {
        key = first;
    }

    public void setSecond(V second) {
        value = second;
    }

    public static <T, E> Pair<T,E> of(T key, E value) {
        return new Pair<T, E>(key, value);
    }

    public static <T, E> Pair<T,E> makePair(T key, E value) {
        return new Pair<T, E>(key, value);
    }

    public static <T, E> Pair<T,E> create(T key, E value) {
        return new Pair<T, E>(key, value);
    }

    public static <T, E> Pair<T,E> pairOf(T key, E value) {
        return new Pair<T, E>(key, value);
    }
}
