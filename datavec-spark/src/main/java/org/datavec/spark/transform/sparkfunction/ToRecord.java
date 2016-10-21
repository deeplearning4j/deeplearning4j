package org.datavec.spark.transform.sparkfunction;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.sql.Row;
import org.datavec.api.writable.Writable;

import java.util.List;

/**
 * Created by agibsonccc on 10/21/16.
 */
public class ToRecord  implements Function<Row,List<Writable>> {

    @Override
    public List<Writable> call(Row v1) throws Exception {
        return null;
    }
}
