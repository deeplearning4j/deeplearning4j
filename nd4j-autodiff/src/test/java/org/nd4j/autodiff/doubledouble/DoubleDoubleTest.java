package org.nd4j.autodiff.doubledouble;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Map.Entry;

import org.junit.Test;

import com.google.common.collect.ImmutableMap;

public class DoubleDoubleTest {

    @Test
    public void should_return_the_correct_value_when_given_value_is_powered_by_two_and_then_powered_by_three()
            throws Exception {
        DoubleDouble givenValue = DoubleDouble.fromString("1.2");
        DoubleDouble two = DoubleDouble.fromString("2");
        DoubleDouble givenValuePoweredByTwo = givenValue.pow(two);
        assertThat(givenValuePoweredByTwo, is(DoubleDouble.fromString("1.44")));

        DoubleDouble three = DoubleDouble.fromString("3");
        DoubleDouble givenValuePoweredByThree = givenValue.pow(three);
        assertThat(givenValuePoweredByThree, is(DoubleDouble.fromString("1.728")));
    }

    @Test
    public void test1() throws Exception {
        DoubleDouble dd = DoubleDouble.fromString("1");
        assertEquals(1d, dd.doubleValue(), 1e-12);
    }

    @Test
    public void test2() throws Exception {
        DoubleDouble dd = DoubleDouble.fromString("1.0");
        assertEquals(1d, dd.doubleValue(), 1e-12);
    }

    @Test
    public void test3() throws Exception {
        DoubleDouble dd = DoubleDouble.fromString("-1.0");
        assertEquals(-1d, dd.doubleValue(), 1e-12);
    }

    @Test
    public void test4() throws Exception {
        DoubleDouble dd = DoubleDouble.fromString("10");
        assertEquals(10d, dd.doubleValue(), 1e-12);
    }

    @Test
    public void test5() throws Exception {
        DoubleDouble dd = DoubleDouble.fromString("-10");
        assertEquals(-10d, dd.doubleValue(), 1e-12);
    }

    @Test
    public void test6() throws Exception {
        DoubleDouble dd = DoubleDouble.fromString("-10");
        assertEquals(-10d, dd.doubleValue(), 1e-12);
    }

    @Test
    public void testToString() throws Exception {
        DoubleDouble dd = DoubleDouble.fromOneDouble(4.75);
        assertEquals("4.75", dd.toString());
    }

    @Test
    public void testEquals() {
        DoubleDouble first = DoubleDouble.fromOneDouble(4.75);
        DoubleDouble second = DoubleDouble.fromOneDouble(4.75);

        assertTrue(first.equals(second));

        first = DoubleDouble.fromOneDouble(4.75458763958447);
        second = DoubleDouble.fromOneDouble(4.75458763958436);

        assertTrue(first.equals(second));
    }

    @Test
    public void testParse() throws Exception {
        ImmutableMap<String, DoubleDouble> testData = ImmutableMap.<String, DoubleDouble> builder()
                .put("1E14", DoubleDouble.fromOneDouble(1E14))
                .put("-1E17", DoubleDouble.fromOneDouble(-1E17))
                .put(" -1E17", DoubleDouble.fromOneDouble(-1E17))
                .put(" 1E14", DoubleDouble.fromOneDouble(1E14))
                .put("+1E14", DoubleDouble.fromOneDouble(1E14))
                .put("1e14", DoubleDouble.fromOneDouble(1E14))
                .put("-1e17", DoubleDouble.fromOneDouble(-1E17))
                .put(" -1e17", DoubleDouble.fromOneDouble(-1E17))
                .put(" 1e14", DoubleDouble.fromOneDouble(1E14))
                .put("+1e14", DoubleDouble.fromOneDouble(1E14))
                .put("1.0000000399999998E-4", DoubleDouble.fromOneDouble(1.0000000399999998E-4))
                .put("3.93460376843724", DoubleDouble.fromOneDouble(3.93460376843724))
                .put("1.0000000299999998E-4", DoubleDouble.fromOneDouble(1.0000000299999998E-4))
                .put("3.9346037077899485", DoubleDouble.fromOneDouble(3.9346037077899485))
                .put("1.0000000199999998E-4", DoubleDouble.fromOneDouble(1.0000000199999998E-4))
                .put("3.934603647142657", DoubleDouble.fromOneDouble(3.934603647142657))
                .put("1.0000000099999999E-4", DoubleDouble.fromOneDouble(1.0000000099999999E-4))
                .put("3.9346035864953652", DoubleDouble.fromOneDouble(3.9346035864953652))
                .build();

        for (Entry<String, DoubleDouble> entry : testData.entrySet()) {
            DoubleDouble actual = DoubleDouble.fromString(entry.getKey());
            DoubleDouble expected = entry.getValue();
            String message = entry.getKey() + " is not equal to " + entry.getValue();
            assertEquals(message, expected, actual);
        }
    }

    @Test
    public void should_parse_5000_string_in_less_than_200_000_ms() {
        int size = 5000;
        String strArray[] = new String[size];
        for (int i = 0; i < size; i++) {
            strArray[i] = Integer.valueOf(i).toString();
        }
        DoubleDouble array[] = new DoubleDouble[size];
        long startTime = getTime();
        for (int i = 0; i < size; i++) {
            array[i] = DoubleDouble.fromString(strArray[i]);
        }
        long endTime = getTime();
        long elapsedTimeInMillis = endTime - startTime;
        System.out.println("elapsedTime : " + elapsedTimeInMillis / 1e9 + " s");
    }

    private long getTime() {
        return System.nanoTime();
    }
}
