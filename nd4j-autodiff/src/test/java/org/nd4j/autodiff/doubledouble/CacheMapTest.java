package org.nd4j.autodiff.doubledouble;

import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class CacheMapTest {

    public static final int KEY_0 = 0;
    public static final double RESULT_A = 1.234;
    public static final double RESULT_B = 2.345;
    public static final int KEY_1 = 1;
    public static final int KEY_2 = 2;
    private CacheMap<Integer, Double> cacheMap;

    @Test
    public void should_return_first_value_on_get_when_sizeLimit_is_not_reached_yet() {
        initCacheMapWithSize(5);
        cacheMap.get(KEY_0, () -> RESULT_A);

        Double getResult = cacheMap.get(KEY_0, () -> RESULT_B);

        assertTrue(getResult == RESULT_A);
    }

    @Test
    public void should_overwrite_previous_values_when_sizeLimit_is_reached() {
        initCacheMapWithSize(2);
        cacheMap.get(KEY_0, () -> RESULT_A);
        cacheMap.get(KEY_1, () -> RESULT_A);
        cacheMap.get(KEY_2, () -> RESULT_A);

        Double getResult = cacheMap.get(KEY_0, () -> RESULT_B);

        assertTrue(getResult == RESULT_B);
    }

    private void initCacheMapWithSize(int sizeLimit) {
        cacheMap = new CacheMap<>(sizeLimit);
    }
}