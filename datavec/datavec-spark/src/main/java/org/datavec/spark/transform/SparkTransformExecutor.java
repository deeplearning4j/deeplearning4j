/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.datavec.spark.transform;

import org.apache.commons.math3.util.Pair;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.datavec.api.transform.DataAction;
import org.datavec.api.transform.Transform;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.filter.Filter;
import org.datavec.api.transform.join.Join;
import org.datavec.api.transform.ops.IAggregableReduceOp;
import org.datavec.api.transform.rank.CalculateSortedRank;
import org.datavec.api.transform.reduce.IAssociativeReducer;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.schema.SequenceSchema;
import org.datavec.api.transform.sequence.ConvertToSequence;
import org.datavec.api.transform.sequence.SequenceSplit;
import org.datavec.api.writable.Writable;
import org.datavec.spark.SequenceEmptyRecordFunction;
import org.datavec.spark.functions.EmptyRecordFunction;
import org.datavec.spark.transform.analysis.SequenceFlatMapFunction;
import org.datavec.spark.transform.filter.SparkFilterFunction;
import org.datavec.spark.transform.join.ExecuteJoinFromCoGroupFlatMapFunction;
import org.datavec.spark.transform.join.ExtractKeysFunction;
import org.datavec.spark.transform.misc.ColumnAsKeyPairFunction;
import org.datavec.spark.transform.rank.UnzipForCalculateSortedRankFunction;
import org.datavec.spark.transform.reduce.MapToPairForReducerFunction;
import org.datavec.spark.transform.sequence.*;
import org.datavec.spark.transform.transform.SequenceSplitFunction;
import org.datavec.spark.transform.transform.SparkTransformFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Tuple2;

import java.util.Comparator;
import java.util.List;

/**
 * Execute a datavec
 * transform process
 * on spark rdds.
 *
 * @author Alex Black
 */
public class SparkTransformExecutor {

    private static final Logger log = LoggerFactory.getLogger(SparkTransformExecutor.class);
    //a boolean jvm argument that when the system property is true
    //will cause some functions to invoke a try catch block and just log errors
    //returning empty records
    public final static String LOG_ERROR_PROPERTY = "org.datavec.spark.transform.logerrors";

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
    public static JavaRDD<List<Writable>> execute(JavaRDD<List<Writable>> inputWritables,
                    TransformProcess transformProcess) {
        if (transformProcess.getFinalSchema() instanceof SequenceSchema) {
            throw new IllegalStateException("Cannot return sequence data with this method");
        }

        return execute(inputWritables, null, transformProcess).getFirst();
    }

    /**
     * Execute the specified TransformProcess with the given input data<br>
     * Note: this method can only be used if the TransformProcess
     * starts with non-sequential data,
     * but returns <it>sequence</it>
     * data (after grouping or converting to a sequence as one of the steps)
     *
     * @param inputWritables   Input data to process
     * @param transformProcess TransformProcess to execute
     * @return Processed (sequence) data
     */
    public static JavaRDD<List<List<Writable>>> executeToSequence(JavaRDD<List<Writable>> inputWritables,
                    TransformProcess transformProcess) {
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
    public static JavaRDD<List<Writable>> executeSequenceToSeparate(JavaRDD<List<List<Writable>>> inputSequence,
                    TransformProcess transformProcess) {
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
    public static JavaRDD<List<List<Writable>>> executeSequenceToSequence(JavaRDD<List<List<Writable>>> inputSequence,
                    TransformProcess transformProcess) {
        if (!(transformProcess.getFinalSchema() instanceof SequenceSchema)) {
            throw new IllegalStateException("Cannot return non-sequence data with this method");
        }

        return execute(null, inputSequence, transformProcess).getSecond();
    }

    /**
     * Returns true if the executor
     * is in try catch mode.
     * @return
     */
    public static boolean isTryCatch() {
        return Boolean.getBoolean(LOG_ERROR_PROPERTY);
    }

    private static Pair<JavaRDD<List<Writable>>, JavaRDD<List<List<Writable>>>> execute(
                    JavaRDD<List<Writable>> inputWritables, JavaRDD<List<List<Writable>>> inputSequence,
                    TransformProcess sequence) {
        JavaRDD<List<Writable>> currentWritables = inputWritables;
        JavaRDD<List<List<Writable>>> currentSequence = inputSequence;

        List<DataAction> dataActions = sequence.getActionList();
        if (inputWritables != null) {
            List<Writable> first = inputWritables.first();
            if (first.size() != sequence.getInitialSchema().numColumns()) {
                throw new IllegalStateException("Input data number of columns (" + first.size()
                                + ") does not match the number of columns for the transform process ("
                                + sequence.getInitialSchema().numColumns() + ")");
            }
        } else {
            List<List<Writable>> firstSeq = inputSequence.first();
            if (firstSeq.size() > 0 && firstSeq.get(0).size() != sequence.getInitialSchema().numColumns()) {
                throw new IllegalStateException("Input sequence data number of columns (" + firstSeq.get(0).size()
                                + ") does not match the number of columns for the transform process ("
                                + sequence.getInitialSchema().numColumns() + ")");
            }
        }


        int count = 1;
        for (DataAction d : dataActions) {
            //log.info("Starting execution of stage {} of {}", count, dataActions.size());     //

            if (d.getTransform() != null) {
                Transform t = d.getTransform();
                if (currentWritables != null) {
                    Function<List<Writable>, List<Writable>> function = new SparkTransformFunction(t);
                    if (isTryCatch())
                        currentWritables = currentWritables.map(function).filter(new EmptyRecordFunction());
                    else
                        currentWritables = currentWritables.map(function);
                } else {
                    Function<List<List<Writable>>, List<List<Writable>>> function =
                                    new SparkSequenceTransformFunction(t);
                    if (isTryCatch())
                        currentSequence = currentSequence.map(function).filter(new SequenceEmptyRecordFunction());
                    else
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
                final ConvertToSequence cts = d.getConvertToSequence();

                if(cts.isSingleStepSequencesMode()){
                    //Edge case: create a sequence from each example, by treating each value as a sequence of length 1
                    currentSequence = currentWritables.map(new ConvertToSequenceLengthOne());
                    currentWritables = null;
                } else {
                    //Standard case: join by key
                    //First: convert to PairRDD
                    Schema schema = cts.getInputSchema();
                    int[] colIdxs = schema.getIndexOfColumns(cts.getKeyColumns());
                    JavaPairRDD<List<Writable>, List<Writable>> withKey =
                            currentWritables.mapToPair(new SparkMapToPairByMultipleColumnsFunction(colIdxs));
                    JavaPairRDD<List<Writable>, Iterable<List<Writable>>> grouped = withKey.groupByKey();

                    //Now: convert to a sequence...
                    currentSequence = grouped.mapValues(new SparkGroupToSequenceFunction(cts.getComparator())).values();
                    currentWritables = null;
                }
            } else if (d.getConvertFromSequence() != null) {
                //Convert from sequence...

                if (currentSequence == null) {
                    throw new IllegalStateException(
                                    "Cannot execute ConvertFromSequence operation: current sequence is null");
                }

                currentWritables = currentSequence.flatMap(new SequenceFlatMapFunction());
                currentSequence = null;
            } else if (d.getSequenceSplit() != null) {
                SequenceSplit sequenceSplit = d.getSequenceSplit();
                if (currentSequence == null)
                    throw new IllegalStateException("Error during execution of SequenceSplit: currentSequence is null");
                currentSequence = currentSequence.flatMap(new SequenceSplitFunction(sequenceSplit));
            } else if (d.getReducer() != null) {
                final IAssociativeReducer reducer = d.getReducer();

                if (currentWritables == null)
                    throw new IllegalStateException("Error during execution of reduction: current writables are null. "
                                    + "Trying to execute a reduce operation on a sequence?");
                JavaPairRDD<String, List<Writable>> pair =
                                currentWritables.mapToPair(new MapToPairForReducerFunction(reducer));


                currentWritables = pair.aggregateByKey(reducer.aggregableReducer(),
                                new Function2<IAggregableReduceOp<List<Writable>, List<Writable>>, List<Writable>, IAggregableReduceOp<List<Writable>, List<Writable>>>() {
                                    @Override
                                    public IAggregableReduceOp<List<Writable>, List<Writable>> call(
                                                    IAggregableReduceOp<List<Writable>, List<Writable>> iAggregableReduceOp,
                                                    List<Writable> writables) throws Exception {
                                        iAggregableReduceOp.accept(writables);
                                        return iAggregableReduceOp;
                                    }
                                },
                                new Function2<IAggregableReduceOp<List<Writable>, List<Writable>>, IAggregableReduceOp<List<Writable>, List<Writable>>, IAggregableReduceOp<List<Writable>, List<Writable>>>() {
                                    @Override
                                    public IAggregableReduceOp<List<Writable>, List<Writable>> call(
                                                    IAggregableReduceOp<List<Writable>, List<Writable>> iAggregableReduceOp,
                                                    IAggregableReduceOp<List<Writable>, List<Writable>> iAggregableReduceOp2)
                                                    throws Exception {
                                        iAggregableReduceOp.combine(iAggregableReduceOp2);
                                        return iAggregableReduceOp;
                                    }
                                })
                                .mapValues(new Function<IAggregableReduceOp<List<Writable>, List<Writable>>, List<Writable>>() {

                                    @Override
                                    public List<Writable> call(
                                                    IAggregableReduceOp<List<Writable>, List<Writable>> listIAggregableReduceOp)
                                                    throws Exception {
                                        return listIAggregableReduceOp.get();
                                    }
                                }).values();

            } else if (d.getCalculateSortedRank() != null) {
                CalculateSortedRank csr = d.getCalculateSortedRank();

                if (currentWritables == null) {
                    throw new IllegalStateException(
                                    "Error during execution of CalculateSortedRank: current writables are null. "
                                                    + "Trying to execute a CalculateSortedRank operation on a sequence? (not currently supported)");
                }

                Comparator<Writable> comparator = csr.getComparator();
                String sortColumn = csr.getSortOnColumn();
                int sortColumnIdx = csr.getInputSchema().getIndexOfColumn(sortColumn);
                boolean ascending = csr.isAscending();
                //NOTE: this likely isn't the most efficient implementation.
                JavaPairRDD<Writable, List<Writable>> pairRDD =
                                currentWritables.mapToPair(new ColumnAsKeyPairFunction(sortColumnIdx));
                pairRDD = pairRDD.sortByKey(comparator, ascending);

                JavaPairRDD<Tuple2<Writable, List<Writable>>, Long> zipped = pairRDD.zipWithIndex();
                currentWritables = zipped.map(new UnzipForCalculateSortedRankFunction());
            } else {
                throw new RuntimeException("Unknown/not implemented action: " + d);
            }

            count++;
        }

        //log.info("Completed {} of {} execution steps", count - 1, dataActions.size());       //Lazy execution means this can be printed before anything has actually happened...

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
    public static JavaRDD<List<Writable>> executeJoin(Join join, JavaRDD<List<Writable>> left,
                    JavaRDD<List<Writable>> right) {

        String[] leftColumnNames = join.getJoinColumnsLeft();
        int[] leftColumnIndexes = new int[leftColumnNames.length];
        for (int i = 0; i < leftColumnNames.length; i++) {
            leftColumnIndexes[i] = join.getLeftSchema().getIndexOfColumn(leftColumnNames[i]);
        }

        JavaPairRDD<List<Writable>, List<Writable>> leftJV = left.mapToPair(new ExtractKeysFunction(leftColumnIndexes));

        String[] rightColumnNames = join.getJoinColumnsRight();
        int[] rightColumnIndexes = new int[rightColumnNames.length];
        for (int i = 0; i < rightColumnNames.length; i++) {
            rightColumnIndexes[i] = join.getRightSchema().getIndexOfColumn(rightColumnNames[i]);
        }

        JavaPairRDD<List<Writable>, List<Writable>> rightJV =
                        right.mapToPair(new ExtractKeysFunction(rightColumnIndexes));

        JavaPairRDD<List<Writable>, Tuple2<Iterable<List<Writable>>, Iterable<List<Writable>>>> cogroupedJV =
                        leftJV.cogroup(rightJV);

        return cogroupedJV.flatMap(new ExecuteJoinFromCoGroupFlatMapFunction(join));
    }
}
