package org.datavec.local.transforms;

import org.datavec.api.records.SequenceRecord;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.analysis.AnalysisCounter;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.analysis.DataVecAnalysisUtils;
import org.datavec.api.transform.analysis.columns.ColumnAnalysis;
import org.datavec.api.transform.analysis.histogram.HistogramCounter;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.analysis.aggregate.AnalysisAddFunction;
import org.datavec.local.transforms.analysis.histogram.HistogramAddFunction;

import java.util.*;

public class AnalyzeLocal {
    private static final int DEFAULT_MAX_HISTOGRAM_BUCKETS = 20;

    /**
     * Analyse the specified data - returns a DataAnalysis object with summary information about each column
     *
     * @param schema Schema for data
     * @param rr     Data to analyze
     * @return DataAnalysis for data
     */
    public static DataAnalysis analyze(Schema schema, RecordReader rr) {
        return analyze(schema, rr, DEFAULT_MAX_HISTOGRAM_BUCKETS);
    }

    /**
     * Analyse the specified data - returns a DataAnalysis object with summary information about each column
     *
     * @param schema Schema for data
     * @param rr     Data to analyze
     * @return DataAnalysis for data
     */
    public static DataAnalysis analyze(Schema schema, RecordReader rr, int maxHistogramBuckets){
        AnalysisAddFunction addFn = new AnalysisAddFunction(schema);
        List<AnalysisCounter> counters = null;
        while(rr.hasNext()){
            counters = addFn.apply(counters, rr.next());
        }

        double[][] minsMaxes = new double[counters.size()][2];

        List<ColumnType> columnTypes = schema.getColumnTypes();
        List<ColumnAnalysis> list = DataVecAnalysisUtils.convertCounters(counters, minsMaxes, columnTypes);


        //Do another pass collecting histogram values:
        List<HistogramCounter> histogramCounters = null;
        HistogramAddFunction add = new HistogramAddFunction(maxHistogramBuckets, schema, minsMaxes);
        if(rr.resetSupported()){
            rr.reset();
            while(rr.hasNext()){
                histogramCounters = add.apply(histogramCounters, rr.next());
            }

            DataVecAnalysisUtils.mergeCounters(list, histogramCounters);
        }

        return new DataAnalysis(schema, list);
    }

    /**
     * Get a list of unique values from the specified columns.
     * For sequence data, use {@link #getUniqueSequence(List, Schema, SequenceRecordReader)}
     *
     * @param columnName    Name of the column to get unique values from
     * @param schema        Data schema
     * @param data          Data to get unique values from
     * @return              List of unique values
     */
    public static Set<Writable> getUnique(String columnName, Schema schema, RecordReader data) {
        int colIdx = schema.getIndexOfColumn(columnName);
        Set<Writable> unique = new HashSet<>();
        while(data.hasNext()){
            List<Writable> next = data.next();
            unique.add(next.get(colIdx));
        }
        return unique;
    }

    /**
     * Get a list of unique values from the specified columns.
     * For sequence data, use {@link #getUniqueSequence(String, Schema, SequenceRecordReader)}
     *
     * @param columnNames   Names of the column to get unique values from
     * @param schema        Data schema
     * @param data          Data to get unique values from
     * @return              List of unique values, for each of the specified columns
     */
    public static Map<String,Set<Writable>> getUnique(List<String> columnNames, Schema schema, RecordReader data){
        Map<String,Set<Writable>> m = new HashMap<>();
        for(String s : columnNames){
            m.put(s, new HashSet<>());
        }
        while(data.hasNext()){
            List<Writable> next = data.next();
            for(String s : columnNames){
                int idx = schema.getIndexOfColumn(s);
                m.get(s).add(next.get(idx));
            }
        }
        return m;
    }

    /**
     * Get a list of unique values from the specified column of a sequence
     *
     * @param columnName      Name of the column to get unique values from
     * @param schema          Data schema
     * @param sequenceData    Sequence data to get unique values from
     * @return
     */
    public static Set<Writable> getUniqueSequence(String columnName, Schema schema,
                                                   SequenceRecordReader sequenceData) {
        int colIdx = schema.getIndexOfColumn(columnName);
        Set<Writable> unique = new HashSet<>();
        while(sequenceData.hasNext()){
            List<List<Writable>> next = sequenceData.sequenceRecord();
            for(List<Writable> step : next){
                unique.add(step.get(colIdx));
            }
        }
        return unique;
    }

    /**
     * Get a list of unique values from the specified columns of a sequence
     *
     * @param columnNames     Name of the columns to get unique values from
     * @param schema          Data schema
     * @param sequenceData    Sequence data to get unique values from
     * @return
     */
    public static Map<String,Set<Writable>> getUniqueSequence(List<String> columnNames, Schema schema,
                                                               SequenceRecordReader sequenceData) {
        Map<String,Set<Writable>> m = new HashMap<>();
        for(String s : columnNames){
            m.put(s, new HashSet<>());
        }
        while(sequenceData.hasNext()){
            List<List<Writable>> next = sequenceData.sequenceRecord();
            for(List<Writable> step : next) {
                for (String s : columnNames) {
                    int idx = schema.getIndexOfColumn(s);
                    m.get(s).add(step.get(idx));
                }
            }
        }
        return m;
    }
}
