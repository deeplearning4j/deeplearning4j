package org.datavec.spark.transform.join;

import org.datavec.api.transform.join.Join;
import org.datavec.api.writable.Writable;
import org.datavec.spark.functions.FlatMapFunctionAdapter;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.List;

/**
 * Execute a join
 *
 * @author Alex Black
 */
public class ExecuteJoinFlatMapFunctionAdapter implements FlatMapFunctionAdapter<Tuple2<List<Writable>,Iterable<JoinValue>>, List<Writable>> {

    private final Join join;

    public ExecuteJoinFlatMapFunctionAdapter(Join join) {
        this.join = join;
    }

    @Override
    public Iterable<List<Writable>> call(Tuple2<List<Writable>, Iterable<JoinValue>> t2) throws Exception {

        List<JoinValue> leftList = new ArrayList<>();
        List<JoinValue> rightList = new ArrayList<>();

        for(JoinValue jv : t2._2()){
            if(jv.isLeft()){
                leftList.add(jv);
            } else {
                rightList.add(jv);
            }
        }

        List<List<Writable>> ret = new ArrayList<>();
        Join.JoinType jt = join.getJoinType();
        switch (jt){
            case Inner:
                //Return records where key columns appear in BOTH
                //So if no values from left OR right: no return values
                for(JoinValue jvl : leftList){
                    for(JoinValue jvr : rightList){
                        List<Writable> joined = join.joinExamples(jvl.getValues(), jvr.getValues());
                        ret.add(joined);
                    }
                }
                break;
            case LeftOuter:
                //Return all records from left, even if no corresponding right value (NullWritable in that case)
                for(JoinValue jvl : leftList){
                    if(rightList.size() == 0){
                        List<Writable> joined = join.joinExamples(jvl.getValues(), null);
                        ret.add(joined);
                    } else {
                        for(JoinValue jvr : rightList){
                            List<Writable> joined = join.joinExamples(jvl.getValues(), jvr.getValues());
                            ret.add(joined);
                        }
                    }
                }
                break;
            case RightOuter:
                //Return all records from right, even if no corresponding left value (NullWritable in that case)
                for(JoinValue jvr : rightList){
                    if(leftList.size() == 0){
                        List<Writable> joined = join.joinExamples(null, jvr.getValues());
                        ret.add(joined);
                    } else {
                        for(JoinValue jvl : leftList){
                            List<Writable> joined = join.joinExamples(jvl.getValues(), jvr.getValues());
                            ret.add(joined);
                        }
                    }
                }
                break;
            case FullOuter:
                //Return all records, even if no corresponding left/right value (NullWritable in that case)
                if(leftList.size() == 0){
                    //Only right values
                    for(JoinValue jvr : rightList){
                        List<Writable> joined = join.joinExamples(null, jvr.getValues());
                        ret.add(joined);
                    }
                } else if(rightList.size() == 0){
                    //Only left values
                    for(JoinValue jvl : leftList){
                        List<Writable> joined = join.joinExamples(jvl.getValues(), null);
                        ret.add(joined);
                    }
                } else {
                    //Records from both left and right
                    for(JoinValue jvl : leftList){
                        for(JoinValue jvr : rightList){
                            List<Writable> joined = join.joinExamples(jvl.getValues(), jvr.getValues());
                            ret.add(joined);
                        }
                    }
                }
                break;
        }

        return ret;
    }
}
