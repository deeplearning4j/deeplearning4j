package org.canova.spark.transform.join;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.PairFunction;
import org.canova.api.writable.Writable;
import io.skymind.echidna.api.join.Join;
import io.skymind.echidna.api.schema.Schema;
import scala.Tuple2;

import java.util.List;

/**
 * Map an example to a Tuple2<String,JoinValue> for use in a {@link Join}
 *
 * @author Alex Black
 */
@AllArgsConstructor
public class MapToJoinValuesFunction implements PairFunction<List<Writable>,String,JoinValue> {

    private boolean left;
    private Join join;

    @Override
    public Tuple2<String, JoinValue> call(List<Writable> writables) throws Exception {

        Schema schema;
        String[] keyColumns;
        if(left){
            schema = join.getLeftSchema();
            keyColumns = join.getKeyColumnsLeft();
        } else {
            schema = join.getRightSchema();
            keyColumns = join.getKeyColumnsRight();
        }

        StringBuilder sb = new StringBuilder();
        boolean first = true;
        for(String key : keyColumns){
            int idx = schema.getIndexOfColumn(key);
            if(!first) sb.append("_");
            sb.append(writables.get(idx).toString());
            first = false;
        }

        return new Tuple2<>(sb.toString(), new JoinValue(left,writables));
    }
}
