package org.deeplearning4j.models.oneHotVector.util;

import service.MapUtils;
import service.SeparatorStringBuilder;

import java.util.*;

public class Histogram<T> {
    private final Map<T, DoubleValue> map;
    private DoubleValue count= new DoubleValue();
    public Histogram() {
        this.map = new HashMap<>();
    }

    public Histogram(Map<T, DoubleValue> initial) {
        this.map = initial;
    }

    public void inc(T k) {
        if (!map.containsKey(k))
            map.put(k, new DoubleValue(0));
        map.get(k).increment();
        count.increment();
    }

    public void inc(T k, int val) {
        if (!map.containsKey(k))
            map.put(k, new DoubleValue(0));
        map.get(k).increment(val);
        count.increment(val);
    }

    public void inc(T k, double val) {
        if (!map.containsKey(k))
            map.put(k, new DoubleValue(0));
        map.get(k).increment(val);
        count.increment(val);
    }
    /**
     * doesn't change the count!!
     */
    public void set(T k, int val) {
        set(k,(double)val);
    }

    public void set(T k, double val) {
        DoubleValue v = map.get(k);
        if (v!=null)
            count.decrement(v.getValue());
        map.put(k, new DoubleValue(val));
        count.increment(val);
    }

    public DoubleValue get(T k) {
        return map.get(k);
    }

    public List<T> getTopKKeys(int k) {
        Map<T, DoubleValue> sortedByValue = MapUtils.sortByValue(map);
        List<T> $ = new ArrayList<>(k);

        for (T key : sortedByValue.keySet()) {
            $.add(key);
            if ($.size() == k) break;
        }
        return $;
    }

    public Histogram<T> getTopK(int k) {
        Map<T, DoubleValue> sortedByValue = MapUtils.sortByValue(map);
        Histogram<T> $ = new Histogram<>();
        for (T key : sortedByValue.keySet()) {
            $.set(key, sortedByValue.get(key).getValue());
            if ($.size() == k) break;
        }
        return $;
    }

    public int size() {
        return map.size();
    }

    public Set<T> keySet() {
        return map.keySet();
    }

    public String toString() {
        return toString("\n", true);
    }

    public String toString(String keySeparator, boolean sortByValue) {
        Map<T, DoubleValue> sortedMap = sortByValue ? MapUtils.sortByValue(map) : map;
        SeparatorStringBuilder ssb = new SeparatorStringBuilder(keySeparator);
        for (T key : sortedMap.keySet()) {
            ssb.append(new StringBuilder(key.toString()).append('\t').append(map.get(key).getValue()));
        }
        return ssb.toString();
    }

    public double avergae() {
        return count.getValue() / map.keySet().size();
    }

    public double count() {
        return count.getValue();
    }

    public void remove(T toRemove) {
        if (!map.containsKey(toRemove))
            return;

        count.decrement(map.get(toRemove).getValue());
        map.remove(toRemove);
    }

    public boolean contians(T k) {
        return map.containsKey(k);
    }
}
