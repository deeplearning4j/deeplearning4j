package org.deeplearning4j.arbiter.optimize.serde;

import org.deeplearning4j.arbiter.optimize.serde.jackson.JacksonReflectionLoader;
import org.deeplearning4j.arbiter.optimize.serde.testclasses.TestInterface1;
import org.deeplearning4j.arbiter.optimize.serde.testclasses.TestInterface2;
import org.deeplearning4j.arbiter.optimize.serde.testclasses.TestInterfaceImpl1;
import org.junit.Test;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by agibsonccc on 3/10/17.
 */
public class JacksonReflectionLoaderTest {

    @Test(expected = IllegalArgumentException.class)
    public void testLoad2() throws Exception {
        assertEquals(2, JacksonReflectionLoader.getImpls(Arrays.<Class<?>>asList(TestInterface2.class)).size());
    }

    @Test
    public void testLoad() throws Exception {
        assertEquals(1, JacksonReflectionLoader.getImpls(Arrays.<Class<?>>asList(TestInterface1.class)).size());
    }

    @Test
    public void testInstantiate() throws Exception {
        ObjectMapper objectMapper = JacksonReflectionLoader.findTypesFor(Arrays.<Class<?>>asList(TestInterface1.class));
        String json = objectMapper.writeValueAsString(new TestInterfaceImpl1());
        TestInterface1 instant = JacksonReflectionLoader.instantiateType(TestInterface1.class, json, objectMapper);
        assertTrue(instant.getClass().equals(TestInterfaceImpl1.class));
    }


}