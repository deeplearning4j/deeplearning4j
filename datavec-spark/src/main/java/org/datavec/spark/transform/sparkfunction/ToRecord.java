package org.datavec.spark.transform.sparkfunction;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.sql.Row;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.*;

import java.util.ArrayList;
import java.util.List;

/**
 * Converts a row to a record
 *
 * @author Adam Gibson
 */
@AllArgsConstructor
public class ToRecord  implements Function<Row,List<Writable>> {
    private Schema schema;

    @Override
    public List<Writable> call(Row v1) throws Exception {
        List<Writable> ret = new ArrayList<>();
        for(int i = 0; i < v1.size(); i++) {
            if(v1.get(i) == null) {
                System.err.println("Warning: Found null entry at index " + i + " adding a double 0.0 writable in its place");
                ret.add(new DoubleWritable(0.0));
            }
            else {
                switch(schema.getType(i)) {
                    case Double: ret.add(new DoubleWritable(v1.getDouble(i))); break;
                    case Float: ret.add(new FloatWritable(v1.getFloat(i))); break;
                    case Integer: ret.add(new IntWritable(v1.getInt(i))); break;
                    case Long: ret.add(new LongWritable(v1.getLong(i))); break;
                    default: throw new IllegalStateException("Illegal type");
                }
            }
        }
        return ret;
    }
}
