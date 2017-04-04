package org.nd4j.autodiff.doubledouble;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

class CacheMap<K, V> {

    private final Map<K, V> map = new ConcurrentHashMap<>();

    private final int sizeLimit;

    CacheMap(int sizeLimit) {
        this.sizeLimit = sizeLimit;
    }

    V get(K key, Supplier<V> supplier) {
        if (map.size() > sizeLimit) {
            map.clear();
        }
        return map.computeIfAbsent(key, k -> supplier.get());
    }
}
