package org.nd4j.etl4j.spark.transform;

import io.skymind.echidna.spark.join.*;
import io.skymind.echidna.spark.sequence.SparkGroupToSequenceFunction;
import io.skymind.echidna.spark.sequence.SparkMapToPairByColumnFunction;
import io.skymind.echidna.spark.sequence.SparkSequenceFilterFunction;
import io.skymind.echidna.spark.sequence.SparkSequenceTransformFunction;
import org.apache.commons.math3.util.Pair;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.canova.api.writable.Writable;
import io.skymind.echidna.api.DataAction;
import io.skymind.echidna.api.Transform;
import io.skymind.echidna.api.TransformProcess;
import org.nd4j.etl4j.api.transform.join.Join;
import org.nd4j.etl4j.api.transform.rank.CalculateSortedRank;
import org.nd4j.etl4j.api.transform.sequence.ConvertToSequence;
import org.nd4j.etl4j.api.transform.filter.Filter;
import org.nd4j.etl4j.api.transform.reduce.IReducer;
import org.nd4j.etl4j.api.transform.schema.Schema;
import org.nd4j.etl4j.api.transform.schema.SequenceSchema;
import org.nd4j.etl4j.api.transform.sequence.SequenceSplit;
import org.nd4j.etl4j.spark.transform.analysis.SequenceFlatMapFunction;
import io.skymind.echidna.spark.misc.ColumnAsKeyPairFunction;
import io.skymind.echidna.spark.rank.UnzipForCalculateSortedRankFunction;
import io.skymind.echidna.spark.transform.SequenceSplitFunction;
import io.skymind.echidna.spark.filter.SparkFilterFunction;
import io.skymind.echidna.spark.reduce.MapToPairForReducerFunction;
import io.skymind.echidna.spark.reduce.ReducerFunction;
import io.skymind.echidna.spark.transform.SparkTransformFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Tuple2;

import java.util.Comparator;
import java.util.List;

public class SparkTransformExecutor {

    private static final Logger log = LoggerFactory.getLogger(SparkTransformExecutor.class);


    public JavaRDD<List<Writable>> execute(JavaRDD<List<Writable>> inputWritables, TransformProcess sequence ) {
        if(sequence.getFinalSchema() instanceof SequenceSchema){
            throw new IllegalStateException("Cannot return sequence data with this method");
        }

        return execute(inputWritables,null,sequence).getFirst();
//        return inputWritables.flatMap(new SparkTransformProcessFunction(sequence));    //Only works if no toSequence or FromSequence ops are in the TransformSequenc...
    }

    public JavaRDD<List<List<Writable>>> executeToSequence(JavaRDD<List<Writable>> inputWritables, TransformProcess sequence ) {
        if(!(sequence.getFinalSchema() instanceof SequenceSchema)){
            throw new IllegalStateException("Cannot return non-sequence data with this method");
        }

        return execute(inputWritables,null,sequence).getSecond();
    }

    public JavaRDD<List<Writable>> executeSequenceToSeparate(JavaRDD<List<List<Writable>>> inputSequence, TransformProcess sequence ) {
        if(sequence.getFinalSchema() instanceof SequenceSchema){
            throw new IllegalStateException("Cannot return sequence data with this method");
        }

        return execute(null,inputSequence,sequence).getFirst();
    }

    public JavaRDD<List<List<Writable>>> executeSequenceToSequence(JavaRDD<List<List<Writable>>> inputSequence, TransformProcess sequence ) {
        if(!(sequence.getFinalSchema() instanceof SequenceSchema)){
            throw new IllegalStateException("Cannot return non-sequence data with this method");
        }

        return execute(null,inputSequence,sequence).getSecond();
    }


    private Pair<JavaRDD<List<Writable>>,JavaRDD<List<List<Writable>>>>
        execute(JavaRDD<List<Writable>> inputWritables, JavaRDD<List<List<Writable>>> inputSequence,
                TransformProcess sequence ){
        JavaRDD<List<Writable>> currentWritables = inputWritables;
        JavaRDD<List<List<Writable>>> currentSequence = inputSequence;

        List<DataAction> list = sequence.getActionList();

        int count = 1;
        for(DataAction d : list){
            log.info("Starting execution of stage {} of {}",count,list.size());

            if(d.getTransform() != null) {
                Transform t = d.getTransform();
                if(currentWritables != null){
                    Function<List<Writable>, List<Writable>> function = new SparkTransformFunction(t);
                    currentWritables = currentWritables.map(function);
                } else {
                    Function<List<List<Writable>>, List<List<Writable>>> function =
                            new SparkSequenceTransformFunction(t);
                    currentSequence = currentSequence.map(function);
                }
            } else if(d.getFilter() != null ){
                //Filter
                Filter f = d.getFilter();
                if(currentWritables != null){
                    currentWritables = currentWritables.filter(new SparkFilterFunction(f));
                } else {
                    currentSequence = currentSequence.filter(new SparkSequenceFilterFunction(f));
                }

            } else if(d.getConvertToSequence() != null) {
                //Convert to a sequence...
                ConvertToSequence cts = d.getConvertToSequence();

                //First: convert to PairRDD
                Schema schema = cts.getInputSchema();
                int colIdx = schema.getIndexOfColumn(cts.getKeyColumn());
                JavaPairRDD<Writable, List<Writable>> withKey = currentWritables.mapToPair(new SparkMapToPairByColumnFunction(colIdx));
                JavaPairRDD<Writable, Iterable<List<Writable>>> grouped = withKey.groupByKey();

                //Now: convert to a sequence...
                currentSequence = grouped.map(new SparkGroupToSequenceFunction(cts.getComparator()));
                currentWritables = null;
            } else if(d.getConvertFromSequence() != null ) {
                //Convert from sequence...

                if(currentSequence == null){
                    throw new IllegalStateException("Cannot execute ConvertFromSequence operation: current sequence is null");
                }

                currentWritables = currentSequence.flatMap(new SequenceFlatMapFunction());
                currentSequence = null;
            } else if(d.getSequenceSplit() != null ) {
                SequenceSplit sequenceSplit = d.getSequenceSplit();
                if(currentSequence == null) throw new IllegalStateException("Error during execution of SequenceSplit: currentSequence is null");
                currentSequence = currentSequence.flatMap(new SequenceSplitFunction(sequenceSplit));
            } else if(d.getReducer() != null) {
                IReducer reducer = d.getReducer();

                if (currentWritables == null)
                    throw new IllegalStateException("Error during execution of reduction: current writables are null. "
                            + "Trying to execute a reduce operation on a sequence?");
                JavaPairRDD<String, List<Writable>> pair = currentWritables.mapToPair(new MapToPairForReducerFunction(reducer));

                currentWritables = pair.groupByKey().map(new ReducerFunction(reducer));
            } else if(d.getCalculateSortedRank() != null ){
                CalculateSortedRank csr = d.getCalculateSortedRank();

                if(currentWritables == null){
                    throw new IllegalStateException("Error during execution of CalculateSortedRank: current writables are null. "
                            + "Trying to execute a CalculateSortedRank operation on a sequenc? (not currently supported)");
                }

                Comparator<Writable> comparator = csr.getComparator();
                String sortColumn = csr.getSortOnColumn();
                int sortColumnIdx = csr.getInputSchema().getIndexOfColumn(sortColumn);
                boolean ascending = csr.isAscending();
                //NOTE: this likely isn't the most efficient implementation.
                JavaPairRDD<Writable,List<Writable>> pairRDD = currentWritables.mapToPair(new ColumnAsKeyPairFunction(sortColumnIdx));
                pairRDD = pairRDD.sortByKey(comparator,ascending);

                JavaPairRDD<Tuple2<Writable,List<Writable>>,Long> zipped = pairRDD.zipWithIndex();
                currentWritables = zipped.map(new UnzipForCalculateSortedRankFunction());
            } else {
                throw new RuntimeException("Unknown/not implemented action: " + d);
            }

            count++;
        }

        log.info("Completed {} of {} execution steps",count-1,list.size());

        return new Pair<>(currentWritables,currentSequence);
    }


    public JavaRDD<List<Writable>> executeJoin(Join join, JavaRDD<List<Writable>> left, JavaRDD<List<Writable>> right){

        //Extract out the keys, then join
        //This gives us a JavaPairRDD<String,JoinValue>
        JavaPairRDD<String,JoinValue> leftJV = left.mapToPair(new MapToJoinValuesFunction(true,join));
        JavaPairRDD<String,JoinValue> rightJV = right.mapToPair(new MapToJoinValuesFunction(false,join));

        //Then merge, collect by key, execute the join. This is essentially an outer join
        JavaPairRDD<String,JoinValue> both = leftJV.union(rightJV);
        JavaPairRDD<String,Iterable<JoinValue>> grouped = both.groupByKey();
        JavaRDD<JoinedValue> joined = grouped.map(new ExecuteJoinFunction(join));

        //Filter out values where we don't have one or the other (i.e., for inner, and left/right joins)
        // and also flatten the JoinedValue -> List<Writable>.
        return joined.flatMap(new FilterAndFlattenJoinedValues(join.getJoinType()));
    }
}
