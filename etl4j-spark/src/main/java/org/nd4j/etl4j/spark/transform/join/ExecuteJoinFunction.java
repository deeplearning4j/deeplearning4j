package org.nd4j.etl4j.spark.transform.join;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.Function;
import org.canova.api.writable.Writable;
import org.nd4j.etl4j.api.transform.join.Join;
import scala.Tuple2;

import java.util.List;

/**
 * Execute a join
 *
 * @author Alex Black
 */
@AllArgsConstructor
public class ExecuteJoinFunction implements Function<Tuple2<String,Iterable<JoinValue>>, JoinedValue> {

    private Join join;

    @Override
    public JoinedValue call(Tuple2<String, Iterable<JoinValue>> t2) throws Exception {

        //Extract values + check we don't have duplicates...
        JoinValue left = null;
        JoinValue right = null;
        for(JoinValue jv : t2._2()){
            if(jv.isLeft()){
                if(left != null){
                    throw new IllegalStateException("Invalid state: found multiple left values in join with key \"" + t2._1() + "\"");
                }
                left = jv;
            } else {
                if(right != null){
                    throw new IllegalStateException("Invalid state: found multiple right values in join with key \"" + t2._1() + "\"");
                }
                right = jv;
            }
        }
        List<Writable> leftList = (left == null ? null : left.getValues());
        List<Writable> rightList = (right == null ? null : right.getValues());
        List<Writable> joined = join.joinExamples(leftList, rightList);

        return new JoinedValue(left != null, right != null, joined);
    }
}
