package org.nd4j.autodiff.doubledouble;

import java.util.function.Supplier;

class DoubleDoubleCache<T> {

    private final DoubleMap<DoubleMap<T>> map = new DoubleMap<>();

    // The function (or bifunction) used to build the supplier should always be the same for a
    // DoubleDoubleCache instance.
    // However, for performance issues (we don't really understand yet), it is faster to provide
    // it at each call.
    public T get(Double hi, Double lo, Supplier<T> supplier) {
        DoubleMap<T> hiMap = map.get(hi, DoubleMap::new);
        return hiMap.get(lo, supplier);
    }
}
