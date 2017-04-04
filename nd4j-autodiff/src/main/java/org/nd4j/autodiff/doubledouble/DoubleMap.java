package org.nd4j.autodiff.doubledouble;

import java.util.function.Supplier;

class DoubleMap<T> {

    public static final int SIZE_LIMIT = 500000;
    private final CacheMap<Double, T> map = new CacheMap<>(SIZE_LIMIT);

    T get(Double key, Supplier<T> supplier) {
        return map.get(key, supplier);
    }
}
