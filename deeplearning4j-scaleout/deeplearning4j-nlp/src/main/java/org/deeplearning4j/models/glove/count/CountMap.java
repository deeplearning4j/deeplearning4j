package org.deeplearning4j.models.glove.count;

import com.google.common.util.concurrent.AtomicDouble;
import lombok.Data;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

/**
 * Drop-in replacement for CounterMap
 *
 * WORK IN PROGRESS, PLEASE DO NOT USE
 *
 * @author raver119@gmail.com
 */
public class CountMap<T extends SequenceElement> {
    private Map<MapEntry<T>, AtomicDouble> backingMap = new HashMap<>();

    public CountMap() {
        // placeholder
    }

    public void incrementCount(T element1, T element2, double weight) {
        MapEntry<T> tempEntry = new MapEntry<>(element1, element2);
        if (backingMap.containsKey(tempEntry)) {
            backingMap.get(tempEntry).addAndGet(weight);
        } else {
            backingMap.put(tempEntry, new AtomicDouble(weight));
        }
    }

    public double getCount(T element1, T element2) {
        MapEntry<T> tempEntry = new MapEntry<>(element1, element2);
        if (backingMap.containsKey(tempEntry)) {
            return backingMap.get(tempEntry).get();
        } else return 0;
    }

    public Iterator<Pair<T, T>> getPairIterator() {
        return new Iterator<Pair<T, T>>() {
            private  Iterator<CountMap.MapEntry<T>> iterator = backingMap.keySet().iterator();

            @Override
            public boolean hasNext() {
                    return iterator.hasNext();
            }

            @Override
            public Pair<T, T> next() {
                MapEntry<T> entry = iterator.next();
                return new Pair<>(entry.getElement1(), entry.getElement2());
            }

            @Override
            public void remove() {
                throw new UnsupportedOperationException("remove() isn't supported here");
            }
        };
    }

    public int size() {
        return backingMap.size();
    }

    @Data
    public static class MapEntry<T extends SequenceElement> {
        private T element1;
        private T element2;

        public MapEntry(T element1, T element2) {
            this.element1 = element1;
            this.element2 = element2;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            MapEntry<?> mapEntry = (MapEntry<?>) o;

            if (element1 != null ? !element1.equals(mapEntry.element1) : mapEntry.element1 != null) return false;
            return element2 != null ? element2.equals(mapEntry.element2) : mapEntry.element2 == null;

        }

        @Override
        public int hashCode() {
            int result = element1 != null ? element1.hashCode() : 0;
            result = 31 * result + (element2 != null ? element2.hashCode() : 0);
            return result;
        }
    }
}