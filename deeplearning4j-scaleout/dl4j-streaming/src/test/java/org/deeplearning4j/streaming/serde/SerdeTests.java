package org.deeplearning4j.streaming.serde;

import org.canova.api.io.data.DoubleWritable;
import org.canova.api.writable.Writable;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

import static org.junit.Assert.*;

/**
 * Created by agibsonccc on 6/10/16.
 */
public class SerdeTests {

    @Test
    public void testRecordSerde() {
        Collection<Collection<Writable>> records = new ArrayList<>();
        records.add(Arrays.asList((Writable) new DoubleWritable(2.0)));
        records.add(Arrays.asList((Writable) new DoubleWritable(2.0)));
        RecordSerializer serializer = new RecordSerializer();
        byte[] bytes = serializer.serialize("",records);
        RecordDeSerializer deSerializer = new RecordDeSerializer();
        Collection<Collection<Writable>> recordsTest = deSerializer.deserialize("",bytes);
        assertEquals(records,recordsTest);
    }

}
