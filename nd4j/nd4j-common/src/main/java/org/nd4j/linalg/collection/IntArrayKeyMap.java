package org.nd4j.linalg.collection;

import com.google.common.primitives.Ints;
import lombok.Getter;
import org.nd4j.base.Preconditions;

import java.util.*;

/**
 * A map for int arrays backed by a {@link java.util.TreeMap}
 * @param <V> the value for the map.
 *
 * @author Adam Gibson
 */
public class IntArrayKeyMap<V> implements Map<int[],V> {

    private Map<IntArray,V> map = new LinkedHashMap<>();

    @Override
    public int size() {
        return map.size();
    }

    @Override
    public boolean isEmpty() {
        return map.isEmpty();
    }

    @Override
    public boolean containsKey(Object o) {
        return map.containsKey(new IntArray((int[]) o));
    }

    @Override
    public boolean containsValue(Object o) {
        return map.containsValue(new IntArray((int[]) o));
    }

    @Override
    public V get(Object o) {
        return map.get(new IntArray((int[]) o));
    }

    @Override
    public V put(int[] ints, V v) {
        return map.put(new IntArray(ints),v);
    }

    @Override
    public V remove(Object o) {
        return map.remove(new IntArray((int[]) o));
    }

    @Override
    public void putAll(Map<? extends int[], ? extends V> map) {
        for(Entry<? extends int[], ? extends V> entry : map.entrySet()) {
            this.map.put(new IntArray(entry.getKey()),entry.getValue());
        }
    }

    @Override
    public void clear() {
        map.clear();
    }

    @Override
    public Set<int[]> keySet() {
        Set<IntArray> intArrays = map.keySet();
        Set<int[]> ret = new LinkedHashSet<>();
        for(IntArray intArray : intArrays)
            ret.add(intArray.backingArray);
        return ret;
    }

    @Override
    public Collection<V> values() {
        return map.values();
    }

    @Override
    public Set<Entry<int[], V>> entrySet() {
        Set<Map.Entry<IntArray,V>> intArrays = map.entrySet();
        Set<Entry<int[], V>> ret = new LinkedHashSet<>();
        for(Map.Entry<IntArray,V> intArray : intArrays) {
            final Map.Entry<IntArray,V> intArray2 = intArray;
            ret.add(new Map.Entry<int[], V>() {
                @Override
                public int[] getKey() {
                    return intArray2.getKey().backingArray;
                }

                @Override
                public V getValue() {
                    return intArray2.getValue();
                }

                @Override
                public V setValue(V v) {
                    return intArray2.setValue(v);
                }
            });
        }
        return ret;
    }


    public static class IntArray implements Comparable<IntArray> {
        @Getter
        private int[] backingArray;

        public IntArray(int[] backingArray) {
            Preconditions.checkNotNull(backingArray,"Backing array must not be null!");
            this.backingArray = Ints.toArray(new LinkedHashSet<>(Ints.asList(backingArray)));
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            IntArray intArray = (IntArray) o;

            return Arrays.equals(intArray.backingArray,backingArray);
        }

        @Override
        public int hashCode() {
            return Arrays.hashCode(backingArray);
        }

        @Override
        public int compareTo(IntArray intArray) {
            if(this.backingArray.length == 0 || intArray.backingArray.length == 0) {
                return 1;
            }

            else if(Arrays.equals(backingArray,intArray.backingArray))
                return 1;

            return Ints.compare(Ints.max(backingArray),Ints.max(intArray.backingArray));
        }
    }


}
