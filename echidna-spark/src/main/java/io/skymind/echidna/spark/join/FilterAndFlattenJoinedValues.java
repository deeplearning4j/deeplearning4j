package io.skymind.echidna.spark.join;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.canova.api.writable.Writable;
import io.skymind.echidna.api.join.Join;

import java.util.Collections;
import java.util.List;

/**
 * Doing two things here:
 * (a) filter out any unnecessary values, and
 * (b) extract the List<Writable> values from the JoinedValue
 *
 * @author Alex Black
 */
@AllArgsConstructor
public class FilterAndFlattenJoinedValues implements FlatMapFunction<JoinedValue,List<Writable>> {

    private final Join.JoinType joinType;

    @Override
    public Iterable<List<Writable>> call(JoinedValue joinedValue) throws Exception {
        boolean keep;
        switch (joinType){
            case Inner:
                //Only keep joined values where we have both left and right
                keep = joinedValue.isHaveLeft() && joinedValue.isHaveRight();
                break;
            case LeftOuter:
                //Keep all values where left is not missing/null
                keep = joinedValue.isHaveLeft();
                break;
            case RightOuter:
                //Keep all values where right is not missing/null
                keep = joinedValue.isHaveRight();
                break;
            case FullOuter:
                //Keep all values
                keep = true;
                break;
            default:
                throw new RuntimeException("Unknown/not implemented join type: " + joinType);
        }

        if(keep){
            return Collections.singletonList(joinedValue.getValues());
        } else {
            return Collections.emptyList();
        }
    }
}
