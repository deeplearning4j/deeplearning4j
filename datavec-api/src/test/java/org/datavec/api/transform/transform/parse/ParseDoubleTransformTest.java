package org.datavec.api.transform.transform.parse;

import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 10/22/16.
 */
public class ParseDoubleTransformTest {
    @Test
    public void testDoubleTransform() {
        List<Writable> record = new ArrayList<>();
        record.add(new Text("0.0"));
        List<Writable> transformed = Arrays.<Writable>asList(new DoubleWritable(0.0));
        assertEquals(transformed, new ParseDoubleTransform().map(record));
    }


}
