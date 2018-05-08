/*-
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

import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.analysis.SequenceDataAnalysis;
import org.datavec.api.transform.analysis.columns.*;
import org.datavec.api.transform.analysis.sequence.SequenceLengthAnalysis;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.quality.DataQualityAnalysis;
import org.datavec.api.transform.quality.columns.ColumnQuality;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.api.writable.comparator.Comparators;
import org.datavec.spark.transform.analysis.AnalysisCounter;
import org.datavec.spark.transform.analysis.SelectColumnFunction;
import org.datavec.spark.transform.analysis.SequenceFlatMapFunction;
import org.datavec.spark.transform.analysis.SequenceLengthFunction;
import org.datavec.spark.transform.analysis.aggregate.AnalysisAddFunction;
import org.datavec.spark.transform.analysis.aggregate.AnalysisCombineFunction;
import org.datavec.spark.transform.analysis.columns.*;
import org.datavec.spark.transform.analysis.histogram.HistogramAddFunction;
import org.datavec.spark.transform.analysis.histogram.HistogramCombineFunction;
import org.datavec.spark.transform.analysis.histogram.HistogramCounter;
import org.datavec.spark.transform.analysis.seqlength.IntToDoubleFunction;
import org.datavec.spark.transform.analysis.seqlength.SequenceLengthAnalysisAddFunction;
import org.datavec.spark.transform.analysis.seqlength.SequenceLengthAnalysisCounter;
import org.datavec.spark.transform.analysis.seqlength.SequenceLengthAnalysisMergeFunction;
import org.datavec.spark.transform.analysis.string.StringAnalysisCounter;
import org.datavec.spark.transform.analysis.unique.UniqueAddFunction;
import org.datavec.spark.transform.analysis.unique.UniqueMergeFunction;
import org.datavec.spark.transform.filter.FilterWritablesBySchemaFunction;
import org.datavec.spark.transform.misc.ColumnToKeyPairTransform;
import org.datavec.spark.transform.misc.SumLongsFunction2;
import org.datavec.spark.transform.misc.comparator.Tuple2Comparator;
import org.datavec.spark.transform.quality.QualityAnalysisAddFunction;
import org.datavec.spark.transform.quality.QualityAnalysisCombineFunction;
import org.datavec.spark.transform.quality.QualityAnalysisState;
import scala.Tuple2;

import java.util.*;

/**
 * AnalizeSpark: static methods for
 * analyzing and
 * processing {@code RDD<List<Writable>>} and {@code RDD<List<List<Writable>>}
 *
 * @author Alex Black
 */
public class AnalyzeSpark {

    public static final int DEFAULT_HISTOGRAM_BUCKETS = 30;

    public static SequenceDataAnalysis analyzeSequence(Schema schema, JavaRDD<List<List<Writable>>> data) {
        return analyzeSequence(schema, data, DEFAULT_HISTOGRAM_BUCKETS);
    }

    /**
     *
     * @param schema
     * @param data
     * @param maxHistogramBuckets
     * @return
     */
    public static SequenceDataAnalysis analyzeSequence(Schema schema, JavaRDD<List<List<Writable>>> data,
                    int maxHistogramBuckets) {
        data.cache();
        JavaRDD<List<Writable>> fmSeq = data.flatMap(new SequenceFlatMapFunction());
        DataAnalysis da = analyze(schema, fmSeq);
        //Analyze the length of the sequences:
        JavaRDD<Integer> seqLengths = data.map(new SequenceLengthFunction());
        seqLengths.cache();
        SequenceLengthAnalysisCounter counter = new SequenceLengthAnalysisCounter();
        counter = seqLengths.aggregate(counter, new SequenceLengthAnalysisAddFunction(),
                        new SequenceLengthAnalysisMergeFunction());

        int max = counter.getMaxLengthSeen();
        int min = counter.getMinLengthSeen();
        int nBuckets = counter.getMaxLengthSeen() - counter.getMinLengthSeen();

        Tuple2<double[], long[]> hist;
        if (max == min) {
            //Edge case that spark doesn't like
            hist = new Tuple2<>(new double[] {min}, new long[] {counter.getCountTotal()});
        } else if (nBuckets < maxHistogramBuckets) {
            JavaDoubleRDD drdd = seqLengths.mapToDouble(new IntToDoubleFunction());
            hist = drdd.histogram(nBuckets);
        } else {
            JavaDoubleRDD drdd = seqLengths.mapToDouble(new IntToDoubleFunction());
            hist = drdd.histogram(maxHistogramBuckets);
        }
        seqLengths.unpersist();


        SequenceLengthAnalysis lengthAnalysis = SequenceLengthAnalysis.builder()
                        .totalNumSequences(counter.getCountTotal()).minSeqLength(counter.getMinLengthSeen())
                        .maxSeqLength(counter.getMaxLengthSeen()).countZeroLength(counter.getCountZeroLength())
                        .countOneLength(counter.getCountOneLength()).meanLength(counter.getMean())
                        .histogramBuckets(hist._1()).histogramBucketCounts(hist._2()).build();

        return new SequenceDataAnalysis(schema, da.getColumnAnalysis(), lengthAnalysis);
    }


    public static DataAnalysis analyze(Schema schema, JavaRDD<List<Writable>> data) {
        return analyze(schema, data, DEFAULT_HISTOGRAM_BUCKETS);
    }

    public static DataAnalysis analyze(Schema schema, JavaRDD<List<Writable>> data, int maxHistogramBuckets) {
        data.cache();
        /*
         * TODO: Some care should be given to add histogramBuckets and histogramBucketCounts to this in the future
         */

        List<ColumnType> columnTypes = schema.getColumnTypes();
        List<AnalysisCounter> counters =
                        data.aggregate(null, new AnalysisAddFunction(schema), new AnalysisCombineFunction());

        double[][] minsMaxes = new double[counters.size()][2];

        int nColumns = schema.numColumns();
        List<ColumnAnalysis> list = new ArrayList<>(nColumns);

        for (int i = 0; i < nColumns; i++) {
            ColumnType ct = columnTypes.get(i);

            switch (ct) {
                case String:
                    StringAnalysisCounter sac = (StringAnalysisCounter) counters.get(i);
                    list.add(new StringAnalysis.Builder().countTotal(sac.getCountTotal())
                                    .minLength(sac.getMinLengthSeen()).maxLength(sac.getMaxLengthSeen())
                                    .meanLength(sac.getMean()).sampleStdevLength(sac.getSampleStdev())
                                    .sampleVarianceLength(sac.getSampleVariance()).build());
                    minsMaxes[i][0] = sac.getMinLengthSeen();
                    minsMaxes[i][1] = sac.getMaxLengthSeen();
                    break;
                case Integer:
                    IntegerAnalysisCounter iac = (IntegerAnalysisCounter) counters.get(i);
                    IntegerAnalysis ia = new IntegerAnalysis.Builder().min(iac.getMinValueSeen())
                                    .max(iac.getMaxValueSeen()).mean(iac.getMean()).sampleStdev(iac.getSampleStdev())
                                    .sampleVariance(iac.getSampleVariance()).countZero(iac.getCountZero())
                                    .countNegative(iac.getCountNegative()).countPositive(iac.getCountPositive())
                                    .countMinValue(iac.getCountMinValue()).countMaxValue(iac.getCountMaxValue())
                                    .countTotal(iac.getCountTotal()).digest(iac.getDigest()).build();
                    list.add(ia);

                    minsMaxes[i][0] = iac.getMinValueSeen();
                    minsMaxes[i][1] = iac.getMaxValueSeen();

                    break;
                case Long:
                    LongAnalysisCounter lac = (LongAnalysisCounter) counters.get(i);

                    LongAnalysis la = new LongAnalysis.Builder().min(lac.getMinValueSeen()).max(lac.getMaxValueSeen())
                                    .mean(lac.getMean()).sampleStdev(lac.getSampleStdev())
                                    .sampleVariance(lac.getSampleVariance()).countZero(lac.getCountZero())
                                    .countNegative(lac.getCountNegative()).countPositive(lac.getCountPositive())
                                    .countMinValue(lac.getCountMinValue()).countMaxValue(lac.getCountMaxValue())
                                    .countTotal(lac.getCountTotal()).digest(lac.getDigest()).build();

                    list.add(la);

                    minsMaxes[i][0] = lac.getMinValueSeen();
                    minsMaxes[i][1] = lac.getMaxValueSeen();

                    break;
                case Double:
                    DoubleAnalysisCounter dac = (DoubleAnalysisCounter) counters.get(i);
                    DoubleAnalysis da = new DoubleAnalysis.Builder().min(dac.getMinValueSeen())
                                    .max(dac.getMaxValueSeen()).mean(dac.getMean()).sampleStdev(dac.getSampleStdev())
                                    .sampleVariance(dac.getSampleVariance()).countZero(dac.getCountZero())
                                    .countNegative(dac.getCountNegative()).countPositive(dac.getCountPositive())
                                    .countMinValue(dac.getCountMinValue()).countMaxValue(dac.getCountMaxValue())
                                    .countNaN(dac.getCountNaN()).digest(dac.getDigest()).countTotal(dac.getCountTotal()).build();
                    list.add(da);

                    minsMaxes[i][0] = dac.getMinValueSeen();
                    minsMaxes[i][1] = dac.getMaxValueSeen();

                    break;
                case Categorical:
                    CategoricalAnalysisCounter cac = (CategoricalAnalysisCounter) counters.get(i);
                    CategoricalAnalysis ca = new CategoricalAnalysis(cac.getCounts());
                    list.add(ca);

                    break;
                case Time:
                    LongAnalysisCounter lac2 = (LongAnalysisCounter) counters.get(i);

                    TimeAnalysis la2 = new TimeAnalysis.Builder().min(lac2.getMinValueSeen())
                                    .max(lac2.getMaxValueSeen()).mean(lac2.getMean()).sampleStdev(lac2.getSampleStdev())
                                    .sampleVariance(lac2.getSampleVariance()).countZero(lac2.getCountZero())
                                    .countNegative(lac2.getCountNegative()).countPositive(lac2.getCountPositive())
                                    .countMinValue(lac2.getCountMinValue()).countMaxValue(lac2.getCountMaxValue())
                                    .countTotal(lac2.getCountTotal()).digest(lac2.getDigest()).build();

                    list.add(la2);

                    minsMaxes[i][0] = lac2.getMinValueSeen();
                    minsMaxes[i][1] = lac2.getMaxValueSeen();

                    break;
                case Bytes:
                    BytesAnalysisCounter bac = (BytesAnalysisCounter) counters.get(i);
                    list.add(new BytesAnalysis.Builder().countTotal(bac.getCountTotal()).build());
                    break;
                case NDArray:
                    NDArrayAnalysisCounter nac = (NDArrayAnalysisCounter) counters.get(i);
                    NDArrayAnalysis nda = nac.toAnalysisObject();
                    list.add(nda);

                    minsMaxes[i][0] = nda.getMinValue();
                    minsMaxes[i][1] = nda.getMaxValue();

                    break;
                default:
                    throw new IllegalStateException("Unknown column type: " + ct);
            }
        }

        List<HistogramCounter> histogramCounters =
                        data.aggregate(null, new HistogramAddFunction(maxHistogramBuckets, schema, minsMaxes),
                                        new HistogramCombineFunction());

        //Merge analysis values and histogram values
        for (int i = 0; i < list.size(); i++) {
            HistogramCounter hc = histogramCounters.get(i);
            ColumnAnalysis ca = list.get(i);
            if (ca instanceof IntegerAnalysis) {
                ((IntegerAnalysis) ca).setHistogramBuckets(hc.getBins());
                ((IntegerAnalysis) ca).setHistogramBucketCounts(hc.getCounts());
            } else if (ca instanceof DoubleAnalysis) {
                ((DoubleAnalysis) ca).setHistogramBuckets(hc.getBins());
                ((DoubleAnalysis) ca).setHistogramBucketCounts(hc.getCounts());
            } else if (ca instanceof LongAnalysis) {
                ((LongAnalysis) ca).setHistogramBuckets(hc.getBins());
                ((LongAnalysis) ca).setHistogramBucketCounts(hc.getCounts());
            } else if (ca instanceof TimeAnalysis) {
                ((TimeAnalysis) ca).setHistogramBuckets(hc.getBins());
                ((TimeAnalysis) ca).setHistogramBucketCounts(hc.getCounts());
            } else if (ca instanceof StringAnalysis) {
                ((StringAnalysis) ca).setHistogramBuckets(hc.getBins());
                ((StringAnalysis) ca).setHistogramBucketCounts(hc.getCounts());
            } else if (ca instanceof NDArrayAnalysis) {
                ((NDArrayAnalysis) ca).setHistogramBuckets(hc.getBins());
                ((NDArrayAnalysis) ca).setHistogramBucketCounts(hc.getCounts());
            }
        }


        return new DataAnalysis(schema, list);
    }

    /**
     * Randomly sample values from a single column
     *
     * @param count         Number of values to sample
     * @param columnName    Name of the column to sample from
     * @param schema        Schema
     * @param data          Data to sample from
     * @return              A list of random samples
     */
    public static List<Writable> sampleFromColumn(int count, String columnName, Schema schema,
                    JavaRDD<List<Writable>> data) {
        int colIdx = schema.getIndexOfColumn(columnName);
        JavaRDD<Writable> ithColumn = data.map(new SelectColumnFunction(colIdx));

        return ithColumn.takeSample(false, count);
    }

    /**
     * Randomly sample values from a single column, in all sequences.
     * Values may be taken from any sequence (i.e., sequence order is not preserved)
     *
     * @param count         Number of values to sample
     * @param columnName    Name of the column to sample from
     * @param schema        Schema
     * @param sequenceData  Data to sample from
     * @return              A list of random samples
     */
    public static List<Writable> sampleFromColumnSequence(int count, String columnName, Schema schema,
                    JavaRDD<List<List<Writable>>> sequenceData) {
        JavaRDD<List<Writable>> flattenedSequence = sequenceData.flatMap(new SequenceFlatMapFunction());
        return sampleFromColumn(count, columnName, schema, flattenedSequence);
    }

    /**
     * Get a list of unique values from the specified columns.
     * For sequence data, use {@link #getUniqueSequence(List, Schema, JavaRDD)}
     *
     * @param columnName    Name of the column to get unique values from
     * @param schema        Data schema
     * @param data          Data to get unique values from
     * @return              List of unique values
     */
    public static List<Writable> getUnique(String columnName, Schema schema, JavaRDD<List<Writable>> data) {
        int colIdx = schema.getIndexOfColumn(columnName);
        JavaRDD<Writable> ithColumn = data.map(new SelectColumnFunction(colIdx));
        return ithColumn.distinct().collect();
    }

    /**
     * Get a list of unique values from the specified column.
     * For sequence data, use {@link #getUniqueSequence(String, Schema, JavaRDD)}
     *
     * @param columnNames   Names of the column to get unique values from
     * @param schema        Data schema
     * @param data          Data to get unique values from
     * @return              List of unique values, for each of the specified columns
     */
    public static Map<String,List<Writable>> getUnique(List<String> columnNames, Schema schema, JavaRDD<List<Writable>> data){
        Map<String,Set<Writable>> m = data.aggregate(null, new UniqueAddFunction(columnNames, schema), new UniqueMergeFunction());
        Map<String,List<Writable>> out = new HashMap<>();
        for(String s : m.keySet()){
            out.put(s, new ArrayList<>(m.get(s)));
        }
        return out;
    }

    /**
     * Get a list of unique values from the specified column of a sequence
     *
     * @param columnName      Name of the column to get unique values from
     * @param schema          Data schema
     * @param sequenceData    Sequence data to get unique values from
     * @return
     */
    public static List<Writable> getUniqueSequence(String columnName, Schema schema,
                    JavaRDD<List<List<Writable>>> sequenceData) {
        JavaRDD<List<Writable>> flattenedSequence = sequenceData.flatMap(new SequenceFlatMapFunction());
        return getUnique(columnName, schema, flattenedSequence);
    }

    /**
     * Get a list of unique values from the specified columns of a sequence
     *
     * @param columnNames     Name of the columns to get unique values from
     * @param schema          Data schema
     * @param sequenceData    Sequence data to get unique values from
     * @return
     */
    public static Map<String,List<Writable>> getUniqueSequence(List<String> columnNames, Schema schema,
                                                   JavaRDD<List<List<Writable>>> sequenceData) {
        JavaRDD<List<Writable>> flattenedSequence = sequenceData.flatMap(new SequenceFlatMapFunction());
        return getUnique(columnNames, schema, flattenedSequence);
    }

    /**
     * Randomly sample a set of examples
     *
     * @param count    Number of samples to generate
     * @param data     Data to sample from
     * @return         Samples
     */
    public static List<List<Writable>> sample(int count, JavaRDD<List<Writable>> data) {
        return data.takeSample(false, count);
    }

    /**
     * Randomly sample a number of sequences from the data
     * @param count    Number of sequences to sample
     * @param data     Data to sample from
     * @return         Sequence samples
     */
    public static List<List<List<Writable>>> sampleSequence(int count, JavaRDD<List<List<Writable>>> data) {
        return data.takeSample(false, count);
    }


    /**
     *
     * @param schema
     * @param data
     * @return
     */
    public static DataQualityAnalysis analyzeQualitySequence(Schema schema, JavaRDD<List<List<Writable>>> data) {
        JavaRDD<List<Writable>> fmSeq = data.flatMap(new SequenceFlatMapFunction());
        return analyzeQuality(schema, fmSeq);
    }


    /**
     *
     * @param schema
     * @param data
     * @return
     */
    public static DataQualityAnalysis analyzeQuality(final Schema schema, final JavaRDD<List<Writable>> data) {
        data.cache();
        int nColumns = schema.numColumns();


        List<ColumnType> columnTypes = schema.getColumnTypes();
        List<QualityAnalysisState> states = data.aggregate(null, new QualityAnalysisAddFunction(schema),
                        new QualityAnalysisCombineFunction());

        List<ColumnQuality> list = new ArrayList<>(nColumns);

        for (QualityAnalysisState qualityState : states) {
            list.add(qualityState.getColumnQuality());
        }

        return new DataQualityAnalysis(schema, list);

    }

    /**
     * Randomly sample a set of invalid values from a specified column.
     * Values are considered invalid according to the Schema / ColumnMetaData
     *
     * @param numToSample    Maximum number of invalid values to sample
     * @param columnName     Same of the column from which to sample invalid values
     * @param schema         Data schema
     * @param data           Data
     * @return               List of invalid examples
     */
    public static List<Writable> sampleInvalidFromColumn(int numToSample, String columnName, Schema schema,
                    JavaRDD<List<Writable>> data) {
        return sampleInvalidFromColumn(numToSample, columnName, schema, data, false);
    }

    /**
     * Randomly sample a set of invalid values from a specified column.
     * Values are considered invalid according to the Schema / ColumnMetaData
     *
     * @param numToSample    Maximum number of invalid values to sample
     * @param columnName     Same of the column from which to sample invalid values
     * @param schema         Data schema
     * @param data           Data
     * @param ignoreMissing  If true: ignore missing values (NullWritable or empty/null string) when sampling. If false: include missing values in sampling
     * @return               List of invalid examples
     */
    public static List<Writable> sampleInvalidFromColumn(int numToSample, String columnName, Schema schema,
                    JavaRDD<List<Writable>> data, boolean ignoreMissing) {
        //First: filter out all valid entries, to leave only invalid entries
        int colIdx = schema.getIndexOfColumn(columnName);
        JavaRDD<Writable> ithColumn = data.map(new SelectColumnFunction(colIdx));

        ColumnMetaData meta = schema.getMetaData(columnName);

        JavaRDD<Writable> invalid = ithColumn.filter(new FilterWritablesBySchemaFunction(meta, false, ignoreMissing));

        return invalid.takeSample(false, numToSample);
    }

    /**
     * Randomly sample a set of invalid values from a specified column, for a sequence data set.
     * Values are considered invalid according to the Schema / ColumnMetaData
     *
     * @param numToSample    Maximum number of invalid values to sample
     * @param columnName     Same of the column from which to sample invalid values
     * @param schema         Data schema
     * @param data           Data
     * @return               List of invalid examples
     */
    public static List<Writable> sampleInvalidFromColumnSequence(int numToSample, String columnName, Schema schema,
                    JavaRDD<List<List<Writable>>> data) {
        JavaRDD<List<Writable>> flattened = data.flatMap(new SequenceFlatMapFunction());
        return sampleInvalidFromColumn(numToSample, columnName, schema, flattened);
    }

    /**
     * Sample the N most frequently occurring values in the specified column
     *
     * @param nMostFrequent    Top N values to sample
     * @param columnName       Name of the column to sample from
     * @param schema           Schema of the data
     * @param data             RDD containing the data
     * @return                 List of the most frequently occurring Writable objects in that column, along with their counts
     */
    public static Map<Writable, Long> sampleMostFrequentFromColumn(int nMostFrequent, String columnName, Schema schema,
                    JavaRDD<List<Writable>> data) {
        int columnIdx = schema.getIndexOfColumn(columnName);

        JavaPairRDD<Writable, Long> keyedByWritable = data.mapToPair(new ColumnToKeyPairTransform(columnIdx));
        JavaPairRDD<Writable, Long> reducedByWritable = keyedByWritable.reduceByKey(new SumLongsFunction2());

        List<Tuple2<Writable, Long>> list =
                        reducedByWritable.takeOrdered(nMostFrequent, new Tuple2Comparator<Writable>(false));

        List<Tuple2<Writable, Long>> sorted = new ArrayList<>(list);
        Collections.sort(sorted, new Tuple2Comparator<Writable>(false));

        Map<Writable, Long> map = new LinkedHashMap<>();
        for (Tuple2<Writable, Long> t2 : sorted) {
            map.put(t2._1(), t2._2());
        }

        return map;
    }

    /**
     * Get the minimum value for the specified column
     *
     * @param allData    All data
     * @param columnName Name of the column to get the minimum value for
     * @param schema     Schema of the data
     * @return           Minimum value for the column
     */
    public static Writable min(JavaRDD<List<Writable>> allData, String columnName, Schema schema){
        int columnIdx = schema.getIndexOfColumn(columnName);
        JavaRDD<Writable> col = allData.map(new SelectColumnFunction(columnIdx));
        return col.min(Comparators.forType(schema.getType(columnName).getWritableType()));
    }

    /**
     * Get the maximum value for the specified column
     *
     * @param allData    All data
     * @param columnName Name of the column to get the minimum value for
     * @param schema     Schema of the data
     * @return           Maximum value for the column
     */
    public static Writable max(JavaRDD<List<Writable>> allData, String columnName, Schema schema){
        int columnIdx = schema.getIndexOfColumn(columnName);
        JavaRDD<Writable> col = allData.map(new SelectColumnFunction(columnIdx));
        return col.max(Comparators.forType(schema.getType(columnName).getWritableType()));
    }

}
