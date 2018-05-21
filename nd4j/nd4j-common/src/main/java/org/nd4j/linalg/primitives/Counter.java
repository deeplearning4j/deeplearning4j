package org.nd4j.linalg.primitives;


import java.io.Serializable;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Simple counter implementation
 * @author
 */
public class Counter<T> implements Serializable {
    private static final long serialVersionUID = 119L;

    protected ConcurrentHashMap<T, AtomicDouble> map = new ConcurrentHashMap<>();
    protected AtomicDouble totalCount = new AtomicDouble(0);
    protected AtomicBoolean dirty = new AtomicBoolean(false);

    public Counter() {

    }

    public double getCount(T element) {
        AtomicDouble t = map.get(element);
        if (t == null)
            return 0.0;

        return t.get();
    }

    public void incrementCount(T element, double inc) {
        AtomicDouble t = map.get(element);
        if (t != null)
            t.addAndGet(inc);
        else {
            map.put(element, new AtomicDouble(inc));
        }

        totalCount.addAndGet(inc);
    }

    /**
     * This method will increment all elements in collection
     *
     * @param elements
     * @param inc
     */
    public void incrementAll(Collection<T> elements, double inc) {
        for (T element: elements) {
            incrementCount(element, inc);
        }
    }

    /**
     * This method will increment counts of this counter by counts from other counter
     * @param other
     */
    public <T2 extends T> void incrementAll(Counter<T2> other) {
        for (T2 element: other.keySet()) {
            double cnt = other.getCount(element);
            incrementCount(element, cnt);
        }
    }

    /**
     * This method returns probability of given element
     *
     * @param element
     * @return
     */
    public double getProbability(T element) {
        if (totalCount() <= 0.0)
            throw new IllegalStateException("Can't calculate probability with empty counter");

        return getCount(element) / totalCount();
    }

    /**
     * This method sets new counter value for given element
     *
     * @param element element to be updated
     * @param count new counter value
     * @return previous value
     */
    public double setCount(T element, double count) {
        AtomicDouble t = map.get(element);
        if (t != null) {
            double val = t.getAndSet(count);
            dirty.set(true);
            return val;
        } else {
            map.put(element, new AtomicDouble(count));
            totalCount.addAndGet(count);
            return 0;
        }

    }

    /**
     * This method returns Set of elements used in this counter
     *
     * @return
     */
    public Set<T> keySet() {
        return map.keySet();
    }

    /**
     * This method returns TRUE if counter has no elements, FALSE otherwise
     *
     * @return
     */
    public boolean isEmpty() {
        return map.size() == 0;
    }

    /**
     * This method returns Set<Entry> of this counter
     * @return
     */
    public Set<Map.Entry<T, AtomicDouble>> entrySet() {
        return map.entrySet();
    }

    /**
     * This method returns List of elements, sorted by their counts
     * @return
     */
    public List<T> keySetSorted() {
        List<T> result = new ArrayList<>();

        PriorityQueue<Pair<T, Double>> pq = asPriorityQueue();
        while (!pq.isEmpty()) {
            result.add(pq.poll().getFirst());
        }

        return result;
    }

    /**
     * This method will apply normalization to counter values and totals.
     */
    public void normalize() {
        for (T key : keySet()) {
            setCount(key, getCount(key) / totalCount.get());
        }

        rebuildTotals();
    }

    protected void rebuildTotals() {
        totalCount.set(0);
        for (T key : keySet()) {
            totalCount.addAndGet(getCount(key));
        }

        dirty.set(false);
    }

    /**
     * This method returns total sum of counter values
     * @return
     */
    public double totalCount() {
        if (dirty.get())
            rebuildTotals();

        return totalCount.get();
    }

    /**
     * This method removes given key from counter
     *
     * @param element
     * @return counter value
     */
    public double removeKey(T element) {
        AtomicDouble v = map.remove(element);
        dirty.set(true);

        if (v != null)
            return v.get();
        else
            return 0.0;
    }

    /**
     * This method returns element with highest counter value
     *
     * @return
     */
    public T argMax() {
        double maxCount = -Double.MAX_VALUE;
        T maxKey = null;
        for (Map.Entry<T, AtomicDouble> entry : map.entrySet()) {
            if (entry.getValue().get() > maxCount || maxKey == null) {
                maxKey = entry.getKey();
                maxCount = entry.getValue().get();
            }
        }
        return maxKey;
    }

    /**
     * This method will remove all elements with counts below given threshold from counter
     * @param threshold
     */
    public void dropElementsBelowThreshold(double threshold) {
        Iterator<T> iterator = keySet().iterator();
        while (iterator.hasNext()) {
            T element  = iterator.next();
            double val = map.get(element).get();
            if (val < threshold) {
                iterator.remove();
                dirty.set(true);
            }
        }

    }

    /**
     * This method checks, if element exist in this counter
     *
     * @param element
     * @return
     */
    public boolean containsElement(T element) {
        return map.containsKey(element);
    }

    /**
     * This method effectively resets counter to empty state
     */
    public void clear() {
        map.clear();
        totalCount.set(0.0);
        dirty.set(false);
    }

    @Override
    public boolean equals(Object o){
        if(!(o instanceof Counter))
            return false;
        Counter c2 = (Counter)o;
        return map.equals(c2.map);
    }

    @Override
    public int hashCode(){
        return map.hashCode();
    }

    /**
     * Returns total number of tracked elements
     *
     * @return
     */
    public int size() {
        return map.size();
    }

    /**
     * This method removes all elements except of top N by counter values
     * @param N
     */
    public void keepTopNElements(int N){
        PriorityQueue<Pair<T, Double>> queue = asPriorityQueue();
        clear();
        for (int e = 0; e < N; e++) {
            Pair<T, Double> pair = queue.poll();
            if (pair != null)
                incrementCount(pair.getFirst(), pair.getSecond());
        }
    }


    public PriorityQueue<Pair<T, Double>> asPriorityQueue() {
        PriorityQueue<Pair<T, Double>> pq = new PriorityQueue<>(Math.max(1,map.size()), new PairComparator());
        for (Map.Entry<T, AtomicDouble> entry : map.entrySet()) {
            pq.add(Pair.create(entry.getKey(), entry.getValue().get()));
        }

        return pq;
    }


    public PriorityQueue<Pair<T, Double>> asReversedPriorityQueue() {
        PriorityQueue<Pair<T, Double>> pq = new PriorityQueue<>(Math.max(1,map.size()), new ReversedPairComparator());
        for (Map.Entry<T, AtomicDouble> entry : map.entrySet()) {
            pq.add(Pair.create(entry.getKey(), entry.getValue().get()));
        }

        return pq;
    }

    public  class PairComparator implements Comparator<Pair<T, Double>> {

        @Override
        public int compare(Pair<T, Double> o1, Pair<T, Double> o2) {
            return Double.compare(o2.value, o1.value);
        }
    }

    public class ReversedPairComparator implements Comparator<Pair<T, Double>> {

        @Override
        public int compare(Pair<T, Double> o1, Pair<T, Double> o2) {
            return Double.compare(o1.value, o2.value);
        }
    }
}
