package org.nd4j.linalg.primitives;


import com.google.common.util.concurrent.AtomicDouble;

import java.io.Serializable;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 *
 * @author raver119@gmail.com
 */
public class CounterMap<F, S> implements Serializable{
    private static final long serialVersionUID = 119L;

    protected Map<F, Counter<S>> maps = new ConcurrentHashMap<>();

    public boolean isEmpty() {
        return maps.isEmpty();
    }

    public boolean isEmpty(F element){
        if (isEmpty())
            return true;

        Counter<S> m = maps.get(element);
        if (m == null)
            return true;
        else
            return m.isEmpty();
    }


    public void incrementAll(CounterMap<F, S> other) {
        for (Map.Entry<F, Counter<S>> entry : other.maps.entrySet()) {
            F key = entry.getKey();
            Counter<S> innerCounter = entry.getValue();
            for (Map.Entry<S, AtomicDouble> innerEntry : innerCounter.entrySet()) {
                S value = innerEntry.getKey();
                incrementCount(key, value, innerEntry.getValue().get());
            }
        }
    }

    public void incrementCount(F first, S second, double inc) {
        Counter<S> counter = maps.get(first);
        if (counter == null) {
            counter = new Counter<S>();
            maps.put(first, counter);
        }

        counter.incrementCount(second, inc);
    }


    public double getCount(F first, S second) {
        Counter<S> counter = maps.get(first);
        if (counter == null)
            return 0.0;

        return counter.getCount(second);
    }


    public double setCount(F first, S second, double value) {
        Counter<S> counter = maps.get(first);
        if (counter == null) {
            counter = new Counter<S>();
            maps.put(first, counter);
        }

        return counter.setCount(second, value);
    }


    public Pair<F, S> argMax() {
        Double maxCount = -Double.MAX_VALUE;
        Pair<F, S> maxKey = null;
        for (Map.Entry<F, Counter<S>> entry : maps.entrySet()) {
            Counter<S> counter = entry.getValue();
            S localMax = counter.argMax();
            if (counter.getCount(localMax) > maxCount || maxKey == null) {
                maxKey = new Pair<>(entry.getKey(), localMax);
                maxCount = counter.getCount(localMax);
            }
        }
        return maxKey;
    }

    public Set<F> keySet() {
        return maps.keySet();
    }

    public Counter<S> getCounter(F first) {
        return maps.get(first);
    }

    public Iterator<Pair<F, S>> getIterator() {
        return new Iterator<Pair<F, S>>() {

            Iterator<F> outerIt;
            Iterator<S> innerIt;
            F curKey;

            {
                outerIt = keySet().iterator();
            }

            private boolean hasInside() {
                if (innerIt == null || !innerIt.hasNext()) {
                    if (!outerIt.hasNext()) {
                        return false;
                    }
                    curKey = outerIt.next();
                    innerIt = getCounter(curKey).keySet().iterator();
                }
                return true;
            }

            public boolean hasNext() {
                return hasInside();
            }

            public Pair<F, S> next() {
                hasInside();
                if (curKey == null)
                    throw new RuntimeException("Outer element can't be null");

                return Pair.makePair(curKey, innerIt.next());
            }

            public void remove() {
                //
            }
        };

    }
}
