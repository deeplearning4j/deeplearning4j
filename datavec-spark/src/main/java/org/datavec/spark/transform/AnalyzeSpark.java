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

import org.apache.spark.api.java.JavaPairRDD;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.analysis.SequenceDataAnalysis;
import org.datavec.api.transform.analysis.columns.*;
import org.datavec.api.transform.quality.columns.*;
import org.datavec.api.transform.metadata.*;
import org.datavec.api.transform.analysis.sequence.SequenceLengthAnalysis;
import org.datavec.api.transform.quality.DataQualityAnalysis;
import org.datavec.api.transform.schema.Schema;
import org.datavec.spark.transform.analysis.SequenceFlatMapFunction;
import org.datavec.spark.transform.analysis.columns.CategoricalAnalysisCounter;
import org.datavec.spark.transform.analysis.histogram.HistogramAddFunction;
import org.datavec.spark.transform.analysis.histogram.HistogramCounter;
import org.datavec.spark.transform.analysis.seqlength.IntToDoubleFunction;
import org.datavec.spark.transform.analysis.seqlength.SequenceLengthAnalysisAddFunction;
import org.datavec.spark.transform.analysis.seqlength.SequenceLengthAnalysisMergeFunction;
import org.datavec.spark.transform.analysis.string.StringAnalysisCounter;
import org.datavec.spark.transform.filter.FilterWritablesBySchemaFunction;
import org.datavec.spark.transform.misc.ColumnToKeyPairTransform;
import org.datavec.spark.transform.misc.SumLongsFunction2;
import org.datavec.spark.transform.misc.comparator.Tuple2Comparator;
import org.datavec.spark.transform.quality.integer.IntegerQualityAddFunction;
import org.datavec.spark.transform.quality.longq.LongQualityMergeFunction;
import org.datavec.spark.transform.quality.string.StringQualityAddFunction;
import org.datavec.spark.transform.quality.time.TimeQualityAddFunction;
import org.datavec.spark.transform.analysis.AnalysisCounter;
import org.datavec.spark.transform.analysis.SelectColumnFunction;
import org.datavec.spark.transform.analysis.SequenceLengthFunction;
import org.datavec.spark.transform.analysis.aggregate.AnalysisAddFunction;
import org.datavec.spark.transform.analysis.aggregate.AnalysisCombineFunction;
import org.datavec.spark.transform.analysis.columns.BytesAnalysisCounter;
import org.datavec.spark.transform.analysis.columns.IntegerAnalysisCounter;
import org.datavec.spark.transform.analysis.columns.LongAnalysisCounter;
import org.datavec.spark.transform.analysis.columns.DoubleAnalysisCounter;
import org.datavec.spark.transform.analysis.histogram.HistogramCombineFunction;
import org.datavec.spark.transform.analysis.seqlength.SequenceLengthAnalysisCounter;
import org.datavec.spark.transform.quality.categorical.CategoricalQualityAddFunction;
import org.datavec.spark.transform.quality.categorical.CategoricalQualityMergeFunction;
import org.datavec.spark.transform.quality.integer.IntegerQualityMergeFunction;
import org.datavec.spark.transform.quality.longq.LongQualityAddFunction;
import org.datavec.spark.transform.quality.real.RealQualityAddFunction;
import org.datavec.spark.transform.quality.real.RealQualityMergeFunction;
import org.datavec.spark.transform.quality.string.StringQualityMergeFunction;
import org.datavec.spark.transform.quality.time.TimeQualityMergeFunction;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaRDD;
import org.datavec.api.writable.Writable;
import scala.Tuple2;

import java.util.*;

/**
 * AnalizeSpark: static methods for analyzing and processing {@code RDD<List<Writable>>} and {@code RDD<List<List<Writable>>}
 *
 * @author Alex Black
 */
public class AnalyzeSpark {

    public static final int DEFAULT_HISTOGRAM_BUCKETS = 30;

    public static SequenceDataAnalysis analyzeSequence(Schema schema, JavaRDD<List<List<Writable>>> data) {
        return analyzeSequence(schema,data,DEFAULT_HISTOGRAM_BUCKETS);
    }

    public static SequenceDataAnalysis analyzeSequence(Schema schema, JavaRDD<List<List<Writable>>> data, int maxHistogramBuckets) {
        data.cache();
        JavaRDD<List<Writable>> fmSeq = data.flatMap(new SequenceFlatMapFunction());
        DataAnalysis da = analyze(schema, fmSeq);
        //Analyze the length of the sequences:
        JavaRDD<Integer> seqLengths = data.map(new SequenceLengthFunction());
        seqLengths.cache();
        SequenceLengthAnalysisCounter counter = new SequenceLengthAnalysisCounter();
        counter = seqLengths.aggregate(counter,new SequenceLengthAnalysisAddFunction(), new SequenceLengthAnalysisMergeFunction());

        int max = counter.getMaxLengthSeen();
        int min = counter.getMinLengthSeen();
        int nBuckets = counter.getMaxLengthSeen() - counter.getMinLengthSeen();

        Tuple2<double[],long[]> hist;
        if(max == min){
            //Edge case that spark doesn't like
            hist = new Tuple2<>(new double[]{min},new long[]{counter.getCountTotal()});
        } else if(nBuckets < maxHistogramBuckets){
            JavaDoubleRDD drdd = seqLengths.mapToDouble(new IntToDoubleFunction());
            hist = drdd.histogram(nBuckets);
        } else {
            JavaDoubleRDD drdd = seqLengths.mapToDouble(new IntToDoubleFunction());
            hist = drdd.histogram(maxHistogramBuckets);
        }
        seqLengths.unpersist();


        SequenceLengthAnalysis lengthAnalysis = SequenceLengthAnalysis.builder()
                .totalNumSequences(counter.getCountTotal())
                .minSeqLength(counter.getMinLengthSeen())
                .maxSeqLength(counter.getMaxLengthSeen())
                .countZeroLength(counter.getCountZeroLength())
                .countOneLength(counter.getCountOneLength())
                .meanLength(counter.getMean())
                .histogramBuckets(hist._1())
                .histogramBucketCounts(hist._2())
                .build();

        return new SequenceDataAnalysis(schema,da.getColumnAnalysis(),lengthAnalysis);
    }


    public static DataAnalysis analyze(Schema schema, JavaRDD<List<Writable>> data) {
        return analyze(schema,data,DEFAULT_HISTOGRAM_BUCKETS);
    }

    public static DataAnalysis analyze(Schema schema, JavaRDD<List<Writable>> data, int maxHistogramBuckets) {
        data.cache();
        /*
        int nColumns = schema.numColumns();
        //This is inefficient, but it's easy to implement. Good enough for now!
        List<ColumnAnalysis> list = new ArrayList<>(nColumns);
        for( int i=0; i<nColumns; i++ ){

            String columnName = schema.getName(i);
            ColumnType type = schema.getType(i);

            JavaRDD<Writable> ithColumn = data.map(new SelectColumnFunction(i));
            ithColumn.cache();

            switch(type){
                case String:

                    ithColumn.cache();
                    long countUnique = ithColumn.distinct().count();

                    JavaDoubleRDD stringLength = ithColumn.mapToDouble(new StringLengthFunction());
                    StatCounter stringLengthStats = stringLength.stats();

                    long min = (int)stringLengthStats.min();
                    long max = (int)stringLengthStats.max();

                    long nBuckets = max-min+1;

                    Tuple2<double[],long[]> hist;
                    if(max == min){
                        //Edge case that spark doesn't like
                        hist = new Tuple2<>(new double[]{min,min},new long[]{stringLengthStats.count()});
                    } else if(nBuckets < maxHistogramBuckets){
                        hist = stringLength.histogram((int)nBuckets);
                    } else {
                        hist = stringLength.histogram(maxHistogramBuckets);
                    }

                    list.add(new StringAnalysis.Builder()
                            .countTotal(stringLengthStats.count())
                            .countUnique(countUnique)
                            .minLength((int)min)
                            .maxLength((int)max)
                            .meanLength(stringLengthStats.mean())
                            .sampleStdevLength(stringLengthStats.sampleStdev())
                            .sampleVarianceLength(stringLengthStats.sampleVariance())
                            .histogramBuckets(hist._1())
                            .histogramBucketCounts(hist._2())
                            .build());

                    break;
                case Integer:
                    JavaDoubleRDD doubleRDD1 = ithColumn.mapToDouble(new WritableToDoubleFunction());
                    StatCounter stats1 = doubleRDD1.stats();

                    //Now: count number of 0, >0, <0

                    IntegerAnalysisCounter counter = new IntegerAnalysisCounter();
                    counter = ithColumn.aggregate(counter,new IntegerAnalysisAddFunction(),new IntegerAnalysisMergeFunction());

                    long min1 = (int)stats1.min();
                    long max1 = (int)stats1.max();

                    long nBuckets1 = max1-min1+1;

                    Tuple2<double[],long[]> hist1;
                    if(max1 == min1){
                        //Edge case that spark doesn't like
                        hist1 = new Tuple2<>(new double[]{min1,min1},new long[]{stats1.count()});
                    } else if(nBuckets1 < maxHistogramBuckets){
                        hist1 = doubleRDD1.histogram((int)nBuckets1);
                    } else {
                        hist1 = doubleRDD1.histogram(maxHistogramBuckets);
                    }

                    IntegerAnalysis ia = new IntegerAnalysis.Builder()
                            .min((int)stats1.min())
                            .max((int)stats1.max())
                            .mean(stats1.mean())
                            .sampleStdev(stats1.sampleStdev())
                            .sampleVariance(stats1.sampleVariance())
                            .countZero(counter.getCountZero())
                            .countNegative(counter.getCountNegative())
                            .countPositive(counter.getCountPositive())
                            .countMinValue(counter.getCountMinValue())
                            .countMaxValue(counter.getCountMaxValue())
                            .countTotal(stats1.count())
                            .histogramBuckets(hist1._1())
                            .histogramBucketCounts(hist1._2()).build();

                    list.add(ia);
                    break;
                case Long:
                    JavaDoubleRDD doubleRDDLong = ithColumn.mapToDouble(new WritableToDoubleFunction());
                    StatCounter statsLong = doubleRDDLong.stats();

                    LongAnalysisCounter counterL = new LongAnalysisCounter();
                    counterL = ithColumn.aggregate(counterL,new LongAnalysisAddFunction(),new LongAnalysisMergeFunction());

                    long minLong = (long)statsLong.min();
                    long maxLong = (long)statsLong.max();

                    long nBucketsLong = maxLong-minLong+1;

                    Tuple2<double[],long[]> histLong;
                    if(maxLong == minLong){
                        //Edge case that spark doesn't like
                        histLong = new Tuple2<>(new double[]{minLong,minLong},new long[]{statsLong.count()});
                    } else if(nBucketsLong < maxHistogramBuckets){
                        histLong = doubleRDDLong.histogram((int)nBucketsLong);
                    } else {
                        histLong = doubleRDDLong.histogram(maxHistogramBuckets);
                    }

                    LongAnalysis la = new LongAnalysis.Builder()
                            .min((long)statsLong.min())
                            .max((long)statsLong.max())
                            .mean(statsLong.mean())
                            .sampleStdev(statsLong.sampleStdev())
                            .sampleVariance(statsLong.sampleVariance())
                            .countZero(counterL.getCountZero())
                            .countNegative(counterL.getCountNegative())
                            .countPositive(counterL.getCountPositive())
                            .countMinValue(counterL.getCountMinValue())
                            .countMaxValue(counterL.getCountMaxValue())
                            .countTotal(statsLong.count())
                            .histogramBuckets(histLong._1())
                            .histogramBucketCounts(histLong._2()).build();

                    list.add(la);

                    break;
                case Double:
                    JavaDoubleRDD doubleRDD = ithColumn.mapToDouble(new WritableToDoubleFunction());
                    StatCounter stats = doubleRDD.stats();

                    DoubleAnalysisCounter counterR = new DoubleAnalysisCounter();
                    counterR = ithColumn.aggregate(counterR,new DoubleAnalysisAddFunction(),new DoubleAnalysisMergeFunction());

                    long min2 = (int)stats.min();
                    long max2 = (int)stats.max();

                    Tuple2<double[],long[]> hist2;
                    if(max2 == min2){
                        //Edge case that spark doesn't like
                        hist2 = new Tuple2<>(new double[]{min2,min2},new long[]{stats.count()});
                    } else {
                        hist2 = doubleRDD.histogram(maxHistogramBuckets);
                    }

                    DoubleAnalysis ra = new DoubleAnalysis.Builder()
                            .min(stats.min())
                            .max(stats.max())
                            .mean(stats.mean())
                            .sampleStdev(stats.sampleStdev())
                            .sampleVariance(stats.sampleVariance())
                            .countZero(counterR.getCountZero())
                            .countNegative(counterR.getCountNegative())
                            .countPositive(counterR.getCountPositive())
                            .countMinValue(counterR.getCountMinValue())
                            .countMaxValue(counterR.getCountMaxValue())
                            .countTotal(stats.count())
                            .histogramBuckets(hist2._1())
                            .histogramBucketCounts(hist2._2()).build();

                    list.add(ra);

//                    list.merge(new DoubleAnalysis(stats.min(),stats.max(),stats.mean(),stats.sampleStdev(),stats.sampleVariance(),
//                            counterR.getCountZero(),counterR.getCountNegative(),counterR.getCountPositive(),stats.count(),
//                            hist2._1(),hist2._2()));
                    break;
                case Categorical:

                    JavaRDD<String> rdd = ithColumn.map(new WritableToStringFunction());
                    Map<String,Long> map = rdd.countByValue();

                    list.add(new CategoricalAnalysis(map));


                    break;
                case Bytes:
                    list.add(new BytesAnalysis.Builder().build());  //TODO
                    break;

                case Time:


                    JavaDoubleRDD doubleRDDLong2 = ithColumn.mapToDouble(new WritableToDoubleFunction());
                    StatCounter statsLong2 = doubleRDDLong2.stats();

                    LongAnalysisCounter counterL2 = new LongAnalysisCounter();
                    counterL2 = ithColumn.aggregate(counterL2,new LongAnalysisAddFunction(),new LongAnalysisMergeFunction());

                    long minLong2 = (long)statsLong2.min();
                    long maxLong2 = (long)statsLong2.max();

                    long nBucketsLong2 = maxLong2-minLong2+1;

                    Tuple2<double[],long[]> histLong2;
                    if(maxLong2 == minLong2){
                        //Edge case that spark doesn't like
                        histLong2 = new Tuple2<>(new double[]{minLong2,minLong2},new long[]{statsLong2.count()});
                    } else if(nBucketsLong2 < maxHistogramBuckets){
                        histLong2 = doubleRDDLong2.histogram((int)nBucketsLong2);
                    } else {
                        histLong2 = doubleRDDLong2.histogram(maxHistogramBuckets);
                    }

                    TimeAnalysis la2 = new TimeAnalysis.Builder()
                            .min((long)statsLong2.min())
                            .max((long)statsLong2.max())
                            .mean(statsLong2.mean())
                            .sampleStdev(statsLong2.sampleStdev())
                            .sampleVariance(statsLong2.sampleVariance())
                            .countZero(counterL2.getCountZero())
                            .countNegative(counterL2.getCountNegative())
                            .countPositive(counterL2.getCountPositive())
                            .countMinValue(counterL2.getCountMinValue())
                            .countMaxValue(counterL2.getCountMaxValue())
                            .countTotal(statsLong2.count())
                            .histogramBuckets(histLong2._1())
                            .histogramBucketCounts(histLong2._2()).build();

                    list.add(la2);

                    break;
                default:
                    throw new IllegalStateException("Unknown/not implemented column type for analysis: " + type);
            }

            ithColumn.unpersist();
        }
        */

        List<ColumnType> columnTypes = schema.getColumnTypes();
        List<AnalysisCounter> counters = data.aggregate(
                null,
                new AnalysisAddFunction(schema),
                new AnalysisCombineFunction());

        double[][] minsMaxes = new double[counters.size()][2];

        int nColumns = schema.numColumns();
        List<ColumnAnalysis> list = new ArrayList<>(nColumns);

        for( int i=0; i<nColumns; i++ ){
            ColumnType ct = columnTypes.get(i);

            switch(ct){
                case String:
                    StringAnalysisCounter sac = (StringAnalysisCounter)counters.get(i);
                    list.add(new StringAnalysis.Builder()
                            .countTotal(sac.getCountTotal())
                            .minLength(sac.getMinLengthSeen())
                            .maxLength(sac.getMaxLengthSeen())
                            .meanLength(((double)sac.getSumLength()) / sac.getCountTotal())
//                            .sampleStdevLength(stringLengthStats.sampleStdev())
//                            .sampleVarianceLength(stringLengthStats.sampleVariance())
                            .build());
                    minsMaxes[i][0] = sac.getMinLengthSeen();
                    minsMaxes[i][1] = sac.getMaxLengthSeen();
                    break;
                case Integer:
                    IntegerAnalysisCounter iac = (IntegerAnalysisCounter)counters.get(i);
                    IntegerAnalysis ia = new IntegerAnalysis.Builder()
                            .min(iac.getMinValueSeen())
                            .max(iac.getMaxValueSeen())
                            .mean(((double)iac.getSum()) / iac.getCountTotal())
                            .sampleStdev(iac.getSampleStdev())
                            .sampleVariance(iac.getSampleVariance())
                            .countZero(iac.getCountZero())
                            .countNegative(iac.getCountNegative())
                            .countPositive(iac.getCountPositive())
                            .countMinValue(iac.getCountMinValue())
                            .countMaxValue(iac.getCountMaxValue())
                            .countTotal(iac.getCountTotal())
                            .build();
                    list.add(ia);

                    minsMaxes[i][0] = iac.getMinValueSeen();
                    minsMaxes[i][1] = iac.getMaxValueSeen();

                    break;
                case Long:
                    LongAnalysisCounter lac = (LongAnalysisCounter)counters.get(i);

                    LongAnalysis la = new LongAnalysis.Builder()
                            .min(lac.getMinValueSeen())
                            .max(lac.getMaxValueSeen())
                            .mean(lac.getSum().doubleValue() / lac.getCountTotal())
                            .sampleStdev(lac.getSampleStdev())
                            .sampleVariance(lac.getSampleVariance())
                            .countZero(lac.getCountZero())
                            .countNegative(lac.getCountNegative())
                            .countPositive(lac.getCountPositive())
                            .countMinValue(lac.getCountMinValue())
                            .countMaxValue(lac.getCountMaxValue())
                            .countTotal(lac.getCountTotal())
                            .build();

                    list.add(la);

                    minsMaxes[i][0] = lac.getMinValueSeen();
                    minsMaxes[i][1] = lac.getMaxValueSeen();

                    break;
                case Double:
                    DoubleAnalysisCounter dac = (DoubleAnalysisCounter)counters.get(i);
                    DoubleAnalysis da = new DoubleAnalysis.Builder()
                            .min(dac.getMinValueSeen())
                            .max(dac.getMaxValueSeen())
                            .mean(dac.getSum() / dac.getCountTotal())
                            .sampleStdev(dac.getSampleStdev())
                            .sampleVariance(dac.getSampleVariance())
                            .countZero(dac.getCountZero())
                            .countNegative(dac.getCountNegative())
                            .countPositive(dac.getCountPositive())
                            .countMinValue(dac.getCountMinValue())
                            .countMaxValue(dac.getCountMaxValue())
                            .countNaN(dac.getCountNaN())
                            .countTotal(dac.getCountTotal())
                            .build();
                    list.add(da);

                    minsMaxes[i][0] = dac.getMinValueSeen();
                    minsMaxes[i][1] = dac.getMaxValueSeen();

                    break;
                case Categorical:
                    CategoricalAnalysisCounter cac = (CategoricalAnalysisCounter)counters.get(i);
                    CategoricalAnalysis ca = new CategoricalAnalysis(cac.getCounts());
                    list.add(ca);

                    break;
                case Time:
                    LongAnalysisCounter lac2 = (LongAnalysisCounter)counters.get(i);

                    TimeAnalysis la2 = new TimeAnalysis.Builder()
                            .min(lac2.getMinValueSeen())
                            .max(lac2.getMaxValueSeen())
                            .mean(lac2.getSum().doubleValue() / lac2.getCountTotal())
                            .sampleStdev(lac2.getSampleStdev())
                            .sampleVariance(lac2.getSampleVariance())
                            .countZero(lac2.getCountZero())
                            .countNegative(lac2.getCountNegative())
                            .countPositive(lac2.getCountPositive())
                            .countMinValue(lac2.getCountMinValue())
                            .countMaxValue(lac2.getCountMaxValue())
                            .countTotal(lac2.getCountTotal())
                            .build();

                    list.add(la2);

                    minsMaxes[i][0] = lac2.getMinValueSeen();
                    minsMaxes[i][1] = lac2.getMaxValueSeen();

                    break;
                case Bytes:
                    BytesAnalysisCounter bac = (BytesAnalysisCounter)counters.get(i);
                    list.add(new BytesAnalysis.Builder()
                            .countTotal(bac.getCountTotal())
                            .build());
                    break;
                default:
                    throw new IllegalStateException("Unknown column type: " + ct);
            }
        }

        List<HistogramCounter> histogramCounters = data.aggregate(
                null,
                new HistogramAddFunction(maxHistogramBuckets,schema,minsMaxes),
                new HistogramCombineFunction());

        //Merge analysis values and histogram values
        for( int i=0; i<list.size(); i++ ){
            HistogramCounter hc = histogramCounters.get(i);
            ColumnAnalysis ca = list.get(i);
            if(ca instanceof IntegerAnalysis){
                ((IntegerAnalysis) ca).setHistogramBuckets(hc.getBins());
                ((IntegerAnalysis) ca).setHistogramBucketCounts(hc.getCounts());
            } else if(ca instanceof DoubleAnalysis){
                ((DoubleAnalysis) ca).setHistogramBuckets(hc.getBins());
                ((DoubleAnalysis) ca).setHistogramBucketCounts(hc.getCounts());
            } else if(ca instanceof LongAnalysis){
                ((LongAnalysis) ca).setHistogramBuckets(hc.getBins());
                ((LongAnalysis) ca).setHistogramBucketCounts(hc.getCounts());
            } else if(ca instanceof TimeAnalysis){
                ((TimeAnalysis) ca).setHistogramBuckets(hc.getBins());
                ((TimeAnalysis) ca).setHistogramBucketCounts(hc.getCounts());
            } else if(ca instanceof StringAnalysis){
                ((StringAnalysis) ca).setHistogramBuckets(hc.getBins());
                ((StringAnalysis) ca).setHistogramBucketCounts(hc.getCounts());
            }
        }


        return new DataAnalysis(schema,list);
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
    public static List<Writable> sampleFromColumn(int count, String columnName, Schema schema, JavaRDD<List<Writable>> data){

        int colIdx = schema.getIndexOfColumn(columnName);
        JavaRDD<Writable> ithColumn = data.map(new SelectColumnFunction(colIdx));

        return ithColumn.takeSample(false,count);
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
    public static List<Writable> sampleFromColumnSequence(int count, String columnName, Schema schema, JavaRDD<List<List<Writable>>> sequenceData){
        JavaRDD<List<Writable>> flattenedSequence = sequenceData.flatMap(new SequenceFlatMapFunction());
        return sampleFromColumn(count, columnName, schema, flattenedSequence);
    }

    /**
     * Get a list of unique values from the specified column.
     * For sequence data, use {@link #getUniqueSequence(String, Schema, JavaRDD)}
     *
     * @param columnName    Name of the column to get unique values from
     * @param schema        Data schema
     * @param data          Data to get unique values from
     * @return              List of unique values
     */
    public static List<Writable> getUnique(String columnName, Schema schema, JavaRDD<List<Writable>> data){
        int colIdx = schema.getIndexOfColumn(columnName);
        JavaRDD<Writable> ithColumn = data.map(new SelectColumnFunction(colIdx));

        return ithColumn.distinct().collect();
    }

    /**
     * Get a list of unique values from the specified column of a sequence
     *
     * @param columnName      Name of the column to get unique values from
     * @param schema          Data schema
     * @param sequenceData    Sequence data to get unique values from
     * @return
     */
    public static List<Writable> getUniqueSequence(String columnName, Schema schema, JavaRDD<List<List<Writable>>> sequenceData){
        JavaRDD<List<Writable>> flattenedSequence = sequenceData.flatMap(new SequenceFlatMapFunction());
        return getUnique(columnName, schema, flattenedSequence);
    }

    /**
     * Randomly sample a set of examples
     *
     * @param count    Number of samples to generate
     * @param data     Data to sample from
     * @return         Samples
     */
    public static List<List<Writable>> sample(int count, JavaRDD<List<Writable>> data){
        return data.takeSample(false,count);
    }

    /**
     * Randomly sample a number of sequences from the data
     * @param count    Number of sequences to sample
     * @param data     Data to sample from
     * @return         Sequence samples
     */
    public static List<List<List<Writable>>> sampleSequence(int count, JavaRDD<List<List<Writable>>> data ){
        return data.takeSample(false,count);
    }





    private static ColumnQuality analyze(ColumnMetaData meta, JavaRDD<Writable> ithColumn){

        switch(meta.getColumnType()){
            case String:
                ithColumn.cache();
                long countUnique = ithColumn.distinct().count();

                StringQuality initialString = new StringQuality();
                StringQuality stringQuality = ithColumn.aggregate(initialString,new StringQualityAddFunction((StringMetaData)meta),new StringQualityMergeFunction());
                return stringQuality.add(new StringQuality(0,0,0,0,0,0,0,0,0,countUnique));
            case Integer:
                IntegerQuality initialInt = new IntegerQuality(0,0,0,0,0);
                return ithColumn.aggregate(initialInt,new IntegerQualityAddFunction((IntegerMetaData)meta),new IntegerQualityMergeFunction());
            case Long:
                LongQuality initialLong = new LongQuality();
                return ithColumn.aggregate(initialLong,new LongQualityAddFunction((LongMetaData)meta),new LongQualityMergeFunction());
            case Double:
                DoubleQuality initialReal = new DoubleQuality();
                return ithColumn.aggregate(initialReal,new RealQualityAddFunction((DoubleMetaData)meta), new RealQualityMergeFunction());
            case Categorical:
                CategoricalQuality initialCat = new CategoricalQuality();
                return ithColumn.aggregate(initialCat,new CategoricalQualityAddFunction((CategoricalMetaData)meta),new CategoricalQualityMergeFunction());
            case Time:
                TimeQuality initTimeQuality = new TimeQuality();
                return ithColumn.aggregate(initTimeQuality, new TimeQualityAddFunction((TimeMetaData)meta), new TimeQualityMergeFunction());
            case Bytes:
                return new BytesQuality();    //TODO
            default:
                throw new RuntimeException("Unknown or not implemented column type: " + meta.getColumnType());
        }
    }

    public static DataQualityAnalysis analyzeQualitySequence(Schema schema, JavaRDD<List<List<Writable>>> data){
        JavaRDD<List<Writable>> fmSeq = data.flatMap(new SequenceFlatMapFunction());
        return analyzeQuality(schema, fmSeq);
    }

    public static DataQualityAnalysis analyzeQuality(Schema schema, JavaRDD<List<Writable>> data){

        data.cache();
        int nColumns = schema.numColumns();

        //This is inefficient, but it's easy to implement. Good enough for now!
        List<ColumnQuality> list = new ArrayList<>(nColumns);

        for( int i=0; i<nColumns; i++ ) {
            ColumnMetaData meta = schema.getMetaData(i);
            JavaRDD<Writable> ithColumn = data.map(new SelectColumnFunction(i));
            list.add(analyze(meta, ithColumn));
        }

        return new DataQualityAnalysis(schema,list);
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
    public static List<Writable> sampleInvalidFromColumn(int numToSample, String columnName, Schema schema, JavaRDD<List<Writable>> data) {
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
    public static List<Writable> sampleInvalidFromColumn(int numToSample, String columnName, Schema schema, JavaRDD<List<Writable>> data, boolean ignoreMissing){
        //First: filter out all valid entries, to leave only invalid entries
        int colIdx = schema.getIndexOfColumn(columnName);
        JavaRDD<Writable> ithColumn = data.map(new SelectColumnFunction(colIdx));

        ColumnMetaData meta = schema.getMetaData(columnName);

        JavaRDD<Writable> invalid = ithColumn.filter(new FilterWritablesBySchemaFunction(meta,false,ignoreMissing));

        return invalid.takeSample(false,numToSample);
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
    public static List<Writable> sampleInvalidFromColumnSequence(int numToSample, String columnName, Schema schema, JavaRDD<List<List<Writable>>> data){
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
    public static Map<Writable,Long> sampleMostFrequentFromColumn(int nMostFrequent, String columnName, Schema schema, JavaRDD<List<Writable>> data){

        int columnIdx = schema.getIndexOfColumn(columnName);

        JavaPairRDD<Writable,Long> keyedByWritable = data.mapToPair(new ColumnToKeyPairTransform(columnIdx));
        JavaPairRDD<Writable,Long> reducedByWritable = keyedByWritable.reduceByKey(new SumLongsFunction2());

        List<Tuple2<Writable,Long>> list = reducedByWritable.takeOrdered(nMostFrequent, new Tuple2Comparator<Writable>(false));

        List<Tuple2<Writable,Long>> sorted = new ArrayList<>(list);
        Collections.sort(sorted, new Tuple2Comparator<Writable>(false));

        Map<Writable,Long> map = new LinkedHashMap<>();
        for(Tuple2<Writable,Long> t2 : sorted){
            map.put(t2._1(), t2._2());
        }

        return map;
    }
}
