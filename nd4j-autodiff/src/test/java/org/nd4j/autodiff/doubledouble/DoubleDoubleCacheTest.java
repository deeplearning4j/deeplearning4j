package org.nd4j.autodiff.doubledouble;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.junit.Before;
import org.junit.Test;

public class DoubleDoubleCacheTest {

    private static final Double HI_A = 1.23456789d;
    private static final Double HI_B = 2.34567891d;
    private static final Double LO_A = 3.45678912d;
    private static final Double LO_B = 4.56789123d;
    private static final Double HI_C = 5.67891234d;
    private static final Double LO_C = 6.78912345d;
    private static final Double RESULT_A = 7.89123456d;
    private static final Double RESULT_B = 8.91234567d;
    private static final Double RESULT_C = 9.12345678d;

    private DoubleDoubleCache<Double> doubleDoubleCacheMap;

    @Before
    public void setUp() {
        doubleDoubleCacheMap = new DoubleDoubleCache<>();
    }

    @Test
    public void should_return_the_value_provided_by_the_supplier_on_first_get() {
        double firstGet = doubleDoubleCacheMap.get(HI_A, LO_A, () -> RESULT_A);

        assertTrue(firstGet == RESULT_A);
    }

    @Test
    public void should_not_evaluate_the_supplier_if_value_already_getted_for_the_same_DoubleDouble() {
        doubleDoubleCacheMap.get(HI_A, LO_A, () -> RESULT_A);
        doubleDoubleCacheMap.get(HI_A, LO_A, () -> {
            fail();
            return RESULT_A;
        });
    }

    @Test
    public void should_return_the_value_provided_by_the_first_supplier_when_get_is_called_with_suppliers_which_return_different_values() {
        doubleDoubleCacheMap.get(HI_A, LO_A, () -> RESULT_A);
        double secondGet = doubleDoubleCacheMap.get(HI_A, LO_A, () -> RESULT_B);

        assertTrue(secondGet == RESULT_A);
    }
}