package org.nd4j.remote.serde;

import lombok.val;
import org.junit.Test;
import org.nd4j.remote.clients.serde.impl.*;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class BasicSerdeTests {
    private final static DoubleArraySerde doubleArraySerde = new DoubleArraySerde();
    private final static FloatArraySerde floatArraySerde = new FloatArraySerde();
    private final static StringSerde stringSerde = new StringSerde();
    private final static IntegerSerde integerSerde = new IntegerSerde();
    private final static FloatSerde floatSerde = new FloatSerde();
    private final static DoubleSerde doubleSerde = new DoubleSerde();
    private final static BooleanSerde booleanSerde = new BooleanSerde();

    @Test
    public void testStringSerde_1() {
        val jvmString = "String with { strange } elements";

        val serialized = stringSerde.serialize(jvmString);
        val deserialized = stringSerde.deserialize(serialized);

        assertEquals(jvmString, deserialized);
    }

    @Test
    public void testFloatArraySerDe_1() {
        val jvmArray = new float[] {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

        val serialized = floatArraySerde.serialize(jvmArray);
        val deserialized = floatArraySerde.deserialize(serialized);

        assertArrayEquals(jvmArray, deserialized, 1e-5f);
    }

    @Test
    public void testDoubleArraySerDe_1() {
        val jvmArray = new double[] {1.0, 2.0, 3.0, 4.0, 5.0};

        val serialized = doubleArraySerde.serialize(jvmArray);
        val deserialized = doubleArraySerde.deserialize(serialized);

        assertArrayEquals(jvmArray, deserialized, 1e-5);
    }

    @Test
    public void testFloatSerde_1() {
        val f = 119.f;

        val serialized = floatSerde.serialize(f);
        val deserialized = floatSerde.deserialize(serialized);

        assertEquals(f, deserialized, 1e-5f);
    }

    @Test
    public void testDoubleSerde_1() {
        val d = 119.;

        val serialized = doubleSerde.serialize(d);
        val deserialized = doubleSerde.deserialize(serialized);

        assertEquals(d, deserialized, 1e-5);
    }

    @Test
    public void testIntegerSerde_1() {
        val f = 119;

        val serialized = integerSerde.serialize(f);
        val deserialized = integerSerde.deserialize(serialized);


        assertEquals(f, deserialized.intValue());
    }

    @Test
    public void testBooleanSerde_1() {
        val f = true;

        val serialized = booleanSerde.serialize(f);
        val deserialized = booleanSerde.deserialize(serialized);


        assertEquals(f, deserialized);
    }
}
