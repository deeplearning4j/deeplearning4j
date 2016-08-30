/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.berkeley;



import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;


/**
 * Maintains counts of (key, value) pairs.  The map is structured so that for
 * every key, one can getFromOrigin a counter over values.  Example usage: keys might be
 * words with values being POS tags, and the count being the number of
 * occurences of that word/tag pair.  The sub-counters returned by
 * getCounter(word) would be count distributions over tags for that word.
 *
 * @author Dan Klein
 */
public class CounterMap<K, V> implements java.io.Serializable {
    private static final long serialVersionUID = 1L;
    MapFactory<V, Double> mf;
    Map<K, Counter<V>> counterMap;
    double defltVal = 0.0;
    private static Logger log = LoggerFactory.getLogger(CounterMap.class);

    public interface CountFunction<V> {
        double count(V v1, V v2);
    }

    /**
     * Build a counter map by iterating pairwise over the list.
     * This assumes that the given pair wise items are
     * the same symmetrically. (The relation at i and i + 1 are the same)
     * It creates a counter map such that the pairs are:
     * count(v1,v2) and count(v2,v1) are the same
     * @param items the items to iterate over
     * @param countFunction the function to count
     * @param <V> the type to count
     * @return the counter map pairwise
     */
    public static <V> CounterMap<V,V> runPairWise(final List<V> items,final CountFunction<V> countFunction) {
        ExecutorService exec = new ThreadPoolExecutor(Runtime.getRuntime().availableProcessors(),
                Runtime.getRuntime().availableProcessors(),
                0L, TimeUnit.MILLISECONDS,
                new LinkedBlockingQueue<Runnable>(), new RejectedExecutionHandler() {
            @Override
            public void rejectedExecution(Runnable r, ThreadPoolExecutor executor) {
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                executor.submit(r);
            }
        });

        final AtomicInteger begin = new AtomicInteger(0);
        final AtomicInteger end = new AtomicInteger(items.size() - 1);
        List<Future<V>> futures = new ArrayList<>();

        final CounterMap<V,V> count = parallelCounterMap();
        for(int i = 0; i < items.size() / 2; i++) {
            futures.add(exec.submit(new Callable<V>() {
                @Override
                public V call() throws Exception {
                    int begin2 = begin.incrementAndGet();
                    int end2 = end.decrementAndGet();
                    V v = items.get(begin2);
                    V v2 = items.get(end2);
                    log.trace("Processing " + "(" + begin2 + "," + end2 + ")");
                    //don't double count
                    if(count.getCount(v,v2) > 0)
                        return v;
                    double cost = countFunction.count(v,v2);
                    count.incrementCount(v,v2,cost);
                    count.incrementCount(v2,v,cost);
                    return v;
                }
            }));
        }

        int futureCount = 0;
        for(Future<V> future : futures) {
            try {
                future.get();
                log.trace("Done with " + futureCount++);
            } catch (InterruptedException e) {
                e.printStackTrace();
            } catch (ExecutionException e) {
                e.printStackTrace();
            }
        }
        exec.shutdown();
        try {
            exec.awaitTermination(1,TimeUnit.MINUTES);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }


        return count;

    }

    /**
     * Returns a thread safe counter map
     * @return
     */
    public static <K,V> CounterMap<K,V> parallelCounterMap() {
        MapFactory<K,Double> factory = new MapFactory<K,Double>() {

            private static final long serialVersionUID = 5447027920163740307L;

            @Override
            public Map<K, Double> buildMap() {
                return new ConcurrentHashMap<>();
            }

        };

        CounterMap<K,V> totalWords = new CounterMap(factory,factory);
        return totalWords;
    }

    protected Counter<V> ensureCounter(K key) {
        Counter<V> valueCounter = counterMap.get(key);
        if (valueCounter == null) {
            valueCounter = buildCounter(mf);
            valueCounter.setDeflt(defltVal);
            counterMap.put(key, valueCounter);
        }
        return valueCounter;
    }

    public Collection<Counter<V>> getCounters() {
        return counterMap.values();
    }

    /**
     * @return
     */
    protected Counter<V> buildCounter(MapFactory<V, Double> mf)
    {
        return new Counter<>(mf);
    }

    /**
     * Returns the keys that have been inserted into this CounterMap.
     */
    public Set<K> keySet() {
        return counterMap.keySet();
    }

    /**
     * Sets the count for a particular (key, value) pair.
     */
    public void setCount(K key, V value, double count) {
        Counter<V> valueCounter = ensureCounter(key);
        valueCounter.setCount(value, count);
    }

//	public void setCount(Pair<K,V> pair) {
//		
//	}

    /**
     * Increments the count for a particular (key, value) pair.
     */
    public void incrementCount(K key, V value, double count) {
        Counter<V> valueCounter = ensureCounter(key);
        valueCounter.incrementCount(value, count);
    }

    /**
     * Gets the count of the given (key, value) entry, or zero if that entry is
     * not present.  Does not createComplex any objects.
     */
    public double getCount(K key, V value) {
        Counter<V> valueCounter = counterMap.get(key);
        if (valueCounter == null) return defltVal;
        return valueCounter.getCount(value);
    }

    /**
     * Gets the sub-counter for the given key.  If there is none, a counter is
     * created for that key, and installed in the CounterMap.  You can, for
     * example, add to the returned empty counter directly (though you shouldn't).
     * This is so whether the key is present or not, modifying the returned
     * counter has the same effect (but don't do it).
     */
    public Counter<V> getCounter(K key) {
        return ensureCounter(key);
    }

    public void incrementAll(Map<K, V> map, double count) {
        for (Map.Entry<K, V> entry : map.entrySet()) {
            incrementCount(entry.getKey(), entry.getValue(), count);
        }
    }

    public void incrementAll(CounterMap<K, V> cMap) {
        for (Map.Entry<K,Counter<V>> entry: cMap.counterMap.entrySet()) {
            K key = entry.getKey();
            Counter<V> innerCounter = entry.getValue();
            for (Map.Entry<V, Double> innerEntry: innerCounter.entrySet()) {
                V value = innerEntry.getKey();
                incrementCount(key,value,innerEntry.getValue());
            }
        }
    }

    /**
     * Gets the total count of the given key, or zero if that key is
     * not present.  Does not createComplex any objects.
     */
    public double getCount(K key) {
        Counter<V> valueCounter = counterMap.get(key);
        if (valueCounter == null) return 0.0;
        return valueCounter.totalCount();
    }

    /**
     * Returns the total of all counts in sub-counters.  This implementation is
     * linear; it recalculates the total each time.
     */
    public double totalCount() {
        double total = 0.0;
        for (Map.Entry<K, Counter<V>> entry : counterMap.entrySet()) {
            Counter<V> counter = entry.getValue();
            total += counter.totalCount();
        }
        return total;
    }

    /**
     * Returns the total number of (key, value) entries in the CounterMap (not
     * their total counts).
     */
    public int totalSize() {
        int total = 0;
        for (Map.Entry<K, Counter<V>> entry : counterMap.entrySet()) {
            Counter<V> counter = entry.getValue();
            total += counter.size();
        }
        return total;
    }

    /**
     * The number of keys in this CounterMap (not the number of key-value entries
     * -- use totalSize() for that)
     */
    public int size() {
        return counterMap.size();
    }

    /**
     * True if there are no entries in the CounterMap (false does not mean
     * totalCount > 0)
     */
    public boolean isEmpty() {
        return size() == 0;
    }

    /**
     * Finds the key with maximum count.  This is a linear operation, and ties are broken arbitrarily.
     *
     * @return a key with minumum count
     */
    public Pair<K, V> argMax() {
        double maxCount = Double.NEGATIVE_INFINITY;
        Pair<K, V> maxKey = null;
        for (Map.Entry<K, Counter<V>> entry : counterMap.entrySet()) {
            Counter<V> counter = entry.getValue();
            V localMax = counter.argMax();
            if (counter.getCount(localMax) > maxCount || maxKey == null) {
                maxKey = new Pair<>(entry.getKey(), localMax);
                maxCount = counter.getCount(localMax);
            }
        }
        return maxKey;
    }


    public String toString(int maxValsPerKey) {
        StringBuilder sb = new StringBuilder("[\n");
        for (Map.Entry<K, Counter<V>> entry : counterMap.entrySet()) {
            sb.append("  ");
            sb.append(entry.getKey());
            sb.append(" -> ");
            sb.append(entry.getValue().toString(maxValsPerKey));
            sb.append("\n");
        }
        sb.append("]");
        return sb.toString();
    }

    @Override
    public String toString() {
        return toString(20);
    }

    public String toString(Collection<String> keyFilter) {
        StringBuilder sb = new StringBuilder("[\n");
        for (Map.Entry<K, Counter<V>> entry : counterMap.entrySet()) {
            String keyString = entry.getKey().toString();
            if (keyFilter != null && !keyFilter.contains(keyString)) {
                continue;
            }
            sb.append("  ");
            sb.append(keyString);
            sb.append(" -> ");
            sb.append(entry.getValue().toString(20));
            sb.append("\n");
        }
        sb.append("]");
        return sb.toString();
    }


    public CounterMap(CounterMap<K,V> cm)
    {
        this();
        incrementAll(cm);
    }

    public CounterMap() {
        this(false);
    }


    public boolean isEqualTo(CounterMap<K,V> map)
    {
        boolean tmp = true;
        CounterMap<K,V> bigger = map.size() > size() ? map : this;
        for (K k : bigger.keySet())
        {
            tmp &= map.getCounter(k).isEqualTo(getCounter(k));
        }
        return tmp;
    }





    public CounterMap(MapFactory<K, Counter<V>> outerMF, MapFactory<V, Double> innerMF) {
        mf = innerMF;
        counterMap = outerMF.buildMap();
    }

    public CounterMap(boolean identityHashMap) {
        this(identityHashMap ? new MapFactory.IdentityHashMapFactory<K, Counter<V>>()
                        : new MapFactory.HashMapFactory<K, Counter<V>>(),
                identityHashMap ? new MapFactory.IdentityHashMapFactory<V, Double>()
                        : new MapFactory.HashMapFactory<V, Double>());
    }

    public static void main(String[] args) {
        CounterMap<String, String> bigramCounterMap = new CounterMap<>();
        bigramCounterMap.incrementCount("people", "run", 1);
        bigramCounterMap.incrementCount("cats", "growl", 2);
        bigramCounterMap.incrementCount("cats", "scamper", 3);
        System.out.println(bigramCounterMap);
        System.out.println("Entries for cats: " + bigramCounterMap.getCounter("cats"));
        System.out.println("Entries for dogs: " + bigramCounterMap.getCounter("dogs"));
        System.out.println("Count of cats scamper: "
                + bigramCounterMap.getCount("cats", "scamper"));
        System.out.println("Count of snakes slither: "
                + bigramCounterMap.getCount("snakes", "slither"));
        System.out.println("Total size: " + bigramCounterMap.totalSize());
        System.out.println("Total count: " + bigramCounterMap.totalCount());
        System.out.println(bigramCounterMap);
    }

    public void normalize() {
        for (K key : keySet()) {
            getCounter(key).normalize();
        }
    }

    public void normalizeWithDiscount(double discount) {
        for (K key : keySet()) {
            Counter<V> ctr = getCounter(key);
            double totalCount = ctr.totalCount();
            for (V value : ctr.keySet()) {
                ctr.setCount(value, (ctr.getCount(value) - discount) / totalCount);
            }
        }
    }

    /**
     * Constructs reverse CounterMap where the count of a pair (k,v)
     * is the count of (v,k) in the current CounterMap
     * @return
     */
    public CounterMap<V,K> invert() {
        CounterMap<V,K> invertCounterMap = new CounterMap<>();
        for (K key: this.keySet()) {
            Counter<V> keyCounts = this.getCounter(key);
            for (V val: keyCounts.keySet()) {
                double count = keyCounts.getCount(val);
                invertCounterMap.setCount(val, key, count);
            }
        }
        return invertCounterMap;
    }

    /**
     * Scale all entries in <code>CounterMap</code>
     * by <code>scaleFactor</code>
     * @param scaleFactor
     */
    public void scale(double scaleFactor) {
        for (K key: keySet()) {
            Counter<V> counts = getCounter(key);
            counts.scale(scaleFactor);
        }
    }

    public boolean containsKey(K key) {
        return counterMap.containsKey(key);
    }

    public Iterator<Pair<K,V>> getPairIterator() {

        class PairIterator implements Iterator<Pair<K,V>> {

            Iterator<K> outerIt ;
            Iterator<V> innerIt ;
            K curKey ;

            public PairIterator() {
                outerIt = keySet().iterator();
            }

            private boolean advance() {
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
                return advance();
            }

            public Pair<K, V> next() {
                advance();
                assert curKey != null;
                return Pair.newPair(curKey, innerIt.next());
            }

            public void remove() {
                // TODO Auto-generated method stub

            }

        }
        return new PairIterator();
    }

    public Set<Map.Entry<K, Counter<V>>> getEntrySet() {
        // TODO Auto-generated method stub
        return counterMap.entrySet();
    }

    public void removeKey(K oldIndex)
    {
        counterMap.remove(oldIndex);
    }

    public void setCounter(K newIndex, Counter<V> counter)
    {
        counterMap.put(newIndex, counter);

    }

    public void setDefault(double defltVal) {
        this.defltVal = defltVal;
        for (Counter<V> vCounter : counterMap.values()) {
            vCounter.setDeflt(defltVal);
        }
    }

}
