/*
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.spark.transform;

import org.datavec.spark.transform.join.*;
import org.datavec.spark.transform.misc.ColumnAsKeyPairFunction;
import org.datavec.spark.transform.reduce.MapToPairForReducerFunction;
import org.datavec.spark.transform.sequence.SparkMapToPairByColumnFunction;
import org.datavec.spark.transform.transform.SequenceSplitFunction;
import org.datavec.spark.transform.sequence.SparkGroupToSequenceFunction;
import org.datavec.spark.transform.sequence.SparkSequenceFilterFunction;
import org.datavec.spark.transform.sequence.SparkSequenceTransformFunction;
import org.apache.commons.math3.util.Pair;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.datavec.api.writable.Writable;
import org.datavec.api.transform.DataAction;
import org.datavec.api.transform.Transform;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.join.Join;
import org.datavec.api.transform.rank.CalculateSortedRank;
import org.datavec.api.transform.sequence.ConvertToSequence;
import org.datavec.api.transform.filter.Filter;
import org.datavec.api.transform.reduce.IReducer;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.schema.SequenceSchema;
import org.datavec.api.transform.sequence.SequenceSplit;
import org.datavec.spark.transform.analysis.SequenceFlatMapFunction;
import org.datavec.spark.transform.rank.UnzipForCalculateSortedRankFunction;
import org.datavec.spark.transform.filter.SparkFilterFunction;
import org.datavec.spark.transform.reduce.ReducerFunction;
import org.datavec.spark.transform.transform.SparkTransformFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Tuple2;

import java.util.Comparator;
import java.util.List;

public class SparkTransformExecutor {

    private static final Logger log = LoggerFactory.getLogger(SparkTransformExecutor.class);

    /**
     * @deprecated Use static methods instead of instance methods on SparkTransformExecutor
     */
    @Deprecated
    public SparkTransformExecutor() {

    }

    /**
     * Execute the specified TransformProcess with the given input data<br>
     * Note: this method can only be used if the TransformProcess returns non-sequence data. For TransformProcesses
     * that return a sequence, use {@link #executeToSequence(JavaRDD, TransformProcess)}
     *
     * @param inputWritables   Input data to process
     * @param transformProcess TransformProcess to execute
     * @return Processed data
     */
    public static JavaRDD<List<Writable>> execute(JavaRDD<List<Writable>> inputWritables, TransformProcess transformProcess) {
        if (transformProcess.getFinalSchema() instanceof SequenceSchema) {
            throw new IllegalStateException("Cannot return sequence data with this method");
        }

        return execute(inputWritables, null, transformProcess).getFirst();
    }

    /**
     * Execute the specified TransformProcess with the given input data<br>
     * Note: this method can only be used if the TransformProcess starts with non-sequential data, but returns <it>sequence</it>
     * data (after grouping or converting to a sequence as one of the steps)
     *
     * @param inputWritables   Input data to process
     * @param transformProcess TransformProcess to execute
     * @return Processed (sequence) data
     */
    public static JavaRDD<List<List<Writable>>> executeToSequence(JavaRDD<List<Writable>> inputWritables, TransformProcess transformProcess) {
        if (!(transformProcess.getFinalSchema() instanceof SequenceSchema)) {
            throw new IllegalStateException("Cannot return non-sequence data with this method");
        }

        return execute(inputWritables, null, transformProcess).getSecond();
    }

    /**
     * Execute the specified TransformProcess with the given <i>sequence</i> input data<br>
     * Note: this method can only be used if the TransformProcess starts with sequence data, but returns <i>non-sequential</i>
     * data (after reducing or converting sequential data to individual examples)
     *
     * @param inputSequence    Input sequence data to process
     * @param transformProcess TransformProcess to execute
     * @return Processed (non-sequential) data
     */
    public static JavaRDD<List<Writable>> executeSequenceToSeparate(JavaRDD<List<List<Writable>>> inputSequence, TransformProcess transformProcess) {
        if (transformProcess.getFinalSchema() instanceof SequenceSchema) {
            throw new IllegalStateException("Cannot return sequence data with this method");
        }

        return execute(null, inputSequence, transformProcess).getFirst();
    }

    /**
     * Execute the specified TransformProcess with the given <i>sequence</i> input data<br>
     * Note: this method can only be used if the TransformProcess starts with sequence data, and also returns sequence data
     *
     * @param inputSequence    Input sequence data to process
     * @param transformProcess TransformProcess to execute
     * @return Processed (non-sequential) data
     */
    public static JavaRDD<List<List<Writable>>> executeSequenceToSequence(JavaRDD<List<List<Writable>>> inputSequence, TransformProcess transformProcess) {
        if (!(transformProcess.getFinalSchema() instanceof SequenceSchema)) {
            throw new IllegalStateException("Cannot return non-sequence data with this method");
        }

        return execute(null, inputSequence, transformProcess).getSecond();
    }


    private static Pair<JavaRDD<List<Writable>>, JavaRDD<List<List<Writable>>>>
    execute(JavaRDD<List<Writable>> inputWritables, JavaRDD<List<List<Writable>>> inputSequence,
            TransformProcess sequence) {
        JavaRDD<List<Writable>> currentWritables = inputWritables;
        JavaRDD<List<List<Writable>>> currentSequence = inputSequence;

        List<DataAction> list = sequence.getActionList();

        int count = 1;
        for (DataAction d : list) {
            log.info("Starting execution of stage {} of {}", count, list.size());

            if (d.getTransform() != null) {
                Transform t = d.getTransform();
                if (currentWritables != null) {
                    Function<List<Writable>, List<Writable>> function = new SparkTransformFunction(t);
                    currentWritables = currentWritables.map(function);
                } else {
                    Function<List<List<Writable>>, List<List<Writable>>> function =
                            new SparkSequenceTransformFunction(t);
                    currentSequence = currentSequence.map(function);
                }
            } else if (d.getFilter() != null) {
                //Filter
                Filter f = d.getFilter();
                if (currentWritables != null) {
                    currentWritables = currentWritables.filter(new SparkFilterFunction(f));
                } else {
                    currentSequence = currentSequence.filter(new SparkSequenceFilterFunction(f));
                }

            } else if (d.getConvertToSequence() != null) {
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
            } else if (d.getConvertFromSequence() != null) {
                //Convert from sequence...

                if (currentSequence == null) {
                    throw new IllegalStateException("Cannot execute ConvertFromSequence operation: current sequence is null");
                }

                currentWritables = currentSequence.flatMap(new SequenceFlatMapFunction());
                currentSequence = null;
            } else if (d.getSequenceSplit() != null) {
                SequenceSplit sequenceSplit = d.getSequenceSplit();
                if (currentSequence == null)
                    throw new IllegalStateException("Error during execution of SequenceSplit: currentSequence is null");
                currentSequence = currentSequence.flatMap(new SequenceSplitFunction(sequenceSplit));
            } else if (d.getReducer() != null) {
                IReducer reducer = d.getReducer();

                if (currentWritables == null)
                    throw new IllegalStateException("Error during execution of reduction: current writables are null. "
                            + "Trying to execute a reduce operation on a sequence?");
                JavaPairRDD<String, List<Writable>> pair = currentWritables.mapToPair(new MapToPairForReducerFunction(reducer));

                currentWritables = pair.groupByKey().map(new ReducerFunction(reducer));
            } else if (d.getCalculateSortedRank() != null) {
                CalculateSortedRank csr = d.getCalculateSortedRank();

                if (currentWritables == null) {
                    throw new IllegalStateException("Error during execution of CalculateSortedRank: current writables are null. "
                            + "Trying to execute a CalculateSortedRank operation on a sequenc? (not currently supported)");
                }

                Comparator<Writable> comparator = csr.getComparator();
                String sortColumn = csr.getSortOnColumn();
                int sortColumnIdx = csr.getInputSchema().getIndexOfColumn(sortColumn);
                boolean ascending = csr.isAscending();
                //NOTE: this likely isn't the most efficient implementation.
                JavaPairRDD<Writable, List<Writable>> pairRDD = currentWritables.mapToPair(new ColumnAsKeyPairFunction(sortColumnIdx));
                pairRDD = pairRDD.sortByKey(comparator, ascending);

                JavaPairRDD<Tuple2<Writable, List<Writable>>, Long> zipped = pairRDD.zipWithIndex();
                currentWritables = zipped.map(new UnzipForCalculateSortedRankFunction());
            } else {
                throw new RuntimeException("Unknown/not implemented action: " + d);
            }

            count++;
        }

        log.info("Completed {} of {} execution steps", count - 1, list.size());

        return new Pair<>(currentWritables, currentSequence);
    }

    /**
     * Execute a join on the specified data
     *
     * @param join  Join to execute
     * @param left  Left data for join
     * @param right Right data for join
     * @return Joined data
     */
    public static JavaRDD<List<Writable>> executeJoin(Join join, JavaRDD<List<Writable>> left, JavaRDD<List<Writable>> right) {

        //Extract out the keys, then join
        //This gives us a JavaPairRDD<String,JoinValue>
        JavaPairRDD<String, JoinValue> leftJV = left.mapToPair(new MapToJoinValuesFunction(true, join));
        JavaPairRDD<String, JoinValue> rightJV = right.mapToPair(new MapToJoinValuesFunction(false, join));

        //Then merge, collect by key, execute the join. This is essentially an outer join
        JavaPairRDD<String, JoinValue> both = leftJV.union(rightJV);
        JavaPairRDD<String, Iterable<JoinValue>> grouped = both.groupByKey();
        JavaRDD<JoinedValue> joined = grouped.map(new ExecuteJoinFunction(join));

        //Filter out values where we don't have one or the other (i.e., for inner, and left/right joins)
        // and also flatten the JoinedValue -> List<Writable>.
        return joined.flatMap(new FilterAndFlattenJoinedValues(join.getJoinType()));
    }
}
