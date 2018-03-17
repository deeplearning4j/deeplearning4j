package org.datavec.local.transforms;

import com.codepoetics.protonpack.Indexed;
import com.codepoetics.protonpack.StreamUtils;
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
import org.datavec.arrow.recordreader.ArrowListWritable;
import org.datavec.local.transforms.analysis.SequenceFlatMapFunction;
import org.datavec.local.transforms.functions.EmptyRecordFunction;
import org.datavec.local.transforms.join.ExecuteJoinFromCoGroupFlatMapFunction;
import org.datavec.local.transforms.join.ExtractKeysFunction;
import org.datavec.local.transforms.misc.ColumnAsKeyPairFunction;
import org.datavec.local.transforms.rank.UnzipForCalculateSortedRankFunction;
import org.datavec.local.transforms.reduce.MapToPairForReducerFunction;
import org.datavec.local.transforms.sequence.*;
import org.datavec.local.transforms.transform.SequenceSplitFunction;
import org.datavec.local.transforms.transform.SparkTransformFunction;
import org.datavec.local.transforms.transform.filter.SparkFilterFunction;
import org.nd4j.linalg.function.Function;
import org.nd4j.linalg.primitives.Pair;

import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

public class ArrowTransformExecutor {
    //a boolean jvm argument that when the system property is true
    //will cause some functions to invoke a try catch block and just log errors
    //returning empty records
    public final static String LOG_ERROR_PROPERTY = "org.datavec.spark.transform.logerrors";

    public static void execute(TransformProcess transformProcess,ArrowListWritable on) {
        List<DataAction> actionList = transformProcess.getActionList();
        for(DataAction dataAction : actionList) {
            if(dataAction.getTransform() != null) {
                Transform transform = dataAction.getTransform();
                String columnName = transform.columnName();
                
            }
        }
    }


    /**
     * Execute the specified TransformProcess with the given input data<br>
     * Note: this method can only be used if the TransformProcess returns non-sequence data. For TransformProcesses
     * that return a sequence, use {@link #executeToSequence(List, TransformProcess)}
     *
     * @param inputWritables   Input data to process
     * @param transformProcess TransformProcess to execute
     * @return Processed data
     */
    public static List<List<Writable>> execute(List<List<Writable>> inputWritables,
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
    public static List<List<List<Writable>>> executeToSequence(List<List<Writable>> inputWritables,
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
    public static List<List<Writable>> executeSequenceToSeparate(List<List<List<Writable>>> inputSequence,
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
    public static List<List<List<Writable>>> executeSequenceToSequence(List<List<List<Writable>>> inputSequence,
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

    private static Pair<List<List<Writable>>, List<List<List<Writable>>>> execute(
            List<List<Writable>> inputWritables, List<List<List<Writable>>> inputSequence,
            TransformProcess sequence) {
        List<List<Writable>> currentWritables = inputWritables;
        List<List<List<Writable>>> currentSequence = inputSequence;

        List<DataAction> dataActions = sequence.getActionList();
        if (inputWritables != null) {
            List<Writable> first = inputWritables.get(0);
            if (first.size() != sequence.getInitialSchema().numColumns()) {
                throw new IllegalStateException("Input data number of columns (" + first.size()
                        + ") does not match the number of columns for the transform process ("
                        + sequence.getInitialSchema().numColumns() + ")");
            }
        } else {
            List<List<Writable>> firstSeq = inputSequence.get(0);
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
                        currentWritables = currentWritables.stream().map(input -> function.apply(input)).filter(input -> new EmptyRecordFunction().apply(input)).collect(Collectors.toList());
                    else
                        currentWritables = currentWritables.stream().map(input -> function.apply(input)).collect(Collectors.toList());
                } else {
                    Function<List<List<Writable>>, List<List<Writable>>> function =
                            new SparkSequenceTransformFunction(t);
                    if (isTryCatch())
                        currentSequence = currentSequence.stream().map(input -> function.apply(input)).filter(input ->
                                new SequenceEmptyRecordFunction().apply(input)).collect(Collectors.toList());
                    else
                        currentSequence = currentSequence.stream().map(input -> function.apply(input)).collect(Collectors.toList());


                }
            } else if (d.getFilter() != null) {
                //Filter
                Filter f = d.getFilter();
                if (currentWritables != null) {
                    currentWritables = currentWritables.stream().filter(input -> new SparkFilterFunction(f).apply(input)).collect(Collectors.toList());
                } else {
                    currentSequence = currentSequence.stream().filter(input -> new SparkSequenceFilterFunction(f).apply(input)).collect(Collectors.toList());
                }

            } else if (d.getConvertToSequence() != null) {
                //Convert to a sequence...
                final ConvertToSequence cts = d.getConvertToSequence();

                if(cts.isSingleStepSequencesMode()){
                    //Edge case: create a sequence from each example, by treating each value as a sequence of length 1
                    currentSequence = currentWritables.stream()
                            .map(input -> new ConvertToSequenceLengthOne().apply(input))
                            .collect(Collectors.toList());
                    currentWritables = null;
                } else {
                    //Standard case: join by key
                    //First: convert to PairRDD
                    Schema schema = cts.getInputSchema();
                    int[] colIdxs = schema.getIndexOfColumns(cts.getKeyColumns());
                    List<org.nd4j.linalg.primitives.Pair<List<Writable>, List<Writable>>> withKey =
                            currentWritables.stream().map(inputSequence2 -> new SparkMapToPairByMultipleColumnsFunction(colIdxs).apply(inputSequence2)).collect(Collectors.toList());
                    Pair<List<Writable>, Iterable<List<Writable>>> grouped = withKey.stream().collect(Collectors.groupingBy(input::get));

                    //Now: convert to a sequence...
                    currentSequence = group.map(new SparkGroupToSequenceFunction(cts.getComparator())).values();
                    currentWritables = null;
                }
            } else if (d.getConvertFromSequence() != null) {
                //Convert from sequence...

                if (currentSequence == null) {
                    throw new IllegalStateException(
                            "Cannot execute ConvertFromSequence operation: current sequence is null");
                }

                currentWritables = currentSequence.stream().flatMap(input -> new SequenceFlatMapFunction().call(input)).collect(Collectors.toList());
                currentSequence = null;
            } else if (d.getSequenceSplit() != null) {
                SequenceSplit sequenceSplit = d.getSequenceSplit();
                if (currentSequence == null)
                    throw new IllegalStateException("Error during execution of SequenceSplit: currentSequence is null");
                currentSequence = currentSequence.stream().flatMap(input -> new SequenceSplitFunction(sequenceSplit).call(input)).collect(Collectors.toList());
            } else if (d.getReducer() != null) {
                final IAssociativeReducer reducer = d.getReducer();

                if (currentWritables == null)
                    throw new IllegalStateException("Error during execution of reduction: current writables are null. "
                            + "Trying to execute a reduce operation on a sequence?");
                List<Pair<String, List<Writable>>> pair =
                        currentWritables.stream().map(input -> new MapToPairForReducerFunction(reducer).apply(input)).collect(Collectors.toList());


                currentWritables = pair.aggregateByKey(reducer.aggregableReducer(),
                        new Function<IAggregableReduceOp<List<Writable>, List<Writable>>, List<Writable>, Pair<IAggregableReduceOp<List<Writable>, List<Writable>>>>() {
                            @Override
                            public IAggregableReduceOp<List<Writable>, List<Writable>> call(
                                    IAggregableReduceOp<List<Writable>, List<Writable>> iAggregableReduceOp,
                                    List<Writable> writables) throws Exception {
                                iAggregableReduceOp.accept(writables);
                                return iAggregableReduceOp;
                            }
                        },
                        new Function<Pair<IAggregableReduceOp<List<Writable>, List<Writable>>,
                                IAggregableReduceOp<Pair<List<Writable>, List<Writable>>, IAggregableReduceOp<List<Writable>, List<Writable>>>() {
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
                List<Pair<Writable, List<Writable>>> pairRDD =
                        currentWritables.stream().map(input -> new ColumnAsKeyPairFunction(sortColumnIdx).apply(input))
                                .collect(Collectors.toList());
                pairRDD = pairRDD.stream().sorted().collect(Collectors.toList());

                List<Indexed<Pair<Writable, List<Writable>>>> zipped = StreamUtils.zipWithIndex(pairRDD.stream()).collect(Collectors.toList());
                currentWritables = zipped.stream().map(new UnzipForCalculateSortedRankFunction()).collect(Collectors.toList());
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
    public static List<List<Writable>> executeJoin(Join join, List<List<Writable>> left,
                                                      List<List<Writable>> right) {

        String[] leftColumnNames = join.getJoinColumnsLeft();
        int[] leftColumnIndexes = new int[leftColumnNames.length];
        for (int i = 0; i < leftColumnNames.length; i++) {
            leftColumnIndexes[i] = join.getLeftSchema().getIndexOfColumn(leftColumnNames[i]);
        }
        List<Pair<List<Writable>, List<Writable>>> leftJV = left.stream().map(input ->
                new ExtractKeysFunction(leftColumnIndexes).apply(input)).collect(Collectors.toList());

        String[] rightColumnNames = join.getJoinColumnsRight();
        int[] rightColumnIndexes = new int[rightColumnNames.length];
        for (int i = 0; i < rightColumnNames.length; i++) {
            rightColumnIndexes[i] = join.getRightSchema().getIndexOfColumn(rightColumnNames[i]);
        }

        List<Pair<List<Writable>, List<Writable>>> rightJV =
                right.stream().map(input -> new ExtractKeysFunction(rightColumnIndexes).apply(input))
                        .collect(Collectors.toList());

        List<Pair<List<Writable>, Pair<Iterable<List<Writable>>, Iterable<List<Writable>>>> cogroupedJV =
                leftJV.stream().cogroup(rightJV);

        return cogroupedJV.stream().flatMap(input -> new ExecuteJoinFromCoGroupFlatMapFunction(join).call(input)).collect(Collectors.toList());
    }


}
