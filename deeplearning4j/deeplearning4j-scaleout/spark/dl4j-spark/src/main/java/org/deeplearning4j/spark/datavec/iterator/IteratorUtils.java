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

package org.deeplearning4j.spark.datavec.iterator;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import scala.Tuple2;

import java.util.*;

/**
 * Utilities for working with RDDs and {@link RecordReaderMultiDataSetIterator}
 *
 * @author Alex Black
 */
public class IteratorUtils {

    /**
     * Apply a single reader {@link RecordReaderMultiDataSetIterator} to a {@code JavaRDD<List<Writable>>}.
     * <b>NOTE</b>: The RecordReaderMultiDataSetIterator <it>must</it> use {@link SparkSourceDummyReader} in place of
     * "real" RecordReader instances
     *
     * @param rdd      RDD with writables
     * @param iterator RecordReaderMultiDataSetIterator with {@link SparkSourceDummyReader} readers
     */
    public static JavaRDD<MultiDataSet> mapRRMDSI(JavaRDD<List<Writable>> rdd, RecordReaderMultiDataSetIterator iterator){
        checkIterator(iterator, 1, 0);
        return mapRRMDSIRecords(rdd.map(new Function<List<Writable>,DataVecRecords>(){
            @Override
            public DataVecRecords call(List<Writable> v1) throws Exception {
                return new DataVecRecords(Collections.singletonList(v1), null);
            }
        }), iterator);
    }

    /**
     * Apply a single sequence reader {@link RecordReaderMultiDataSetIterator} to sequence data, in the form of
     * {@code JavaRDD<List<List<Writable>>>}.
     * <b>NOTE</b>: The RecordReaderMultiDataSetIterator <it>must</it> use {@link SparkSourceDummySeqReader} in place of
     * "real" SequenceRecordReader instances
     *
     * @param rdd      RDD with writables
     * @param iterator RecordReaderMultiDataSetIterator with {@link SparkSourceDummySeqReader} sequence readers
     */
    public static JavaRDD<MultiDataSet> mapRRMDSISeq(JavaRDD<List<List<Writable>>> rdd, RecordReaderMultiDataSetIterator iterator){
        checkIterator(iterator, 0, 1);
        return mapRRMDSIRecords(rdd.map(new Function<List<List<Writable>>,DataVecRecords>(){
            @Override
            public DataVecRecords call(List<List<Writable>> v1) throws Exception {
                return new DataVecRecords(null, Collections.singletonList(v1));
            }
        }), iterator);
    }

    /**
     * Apply to an arbitrary mix of non-sequence and sequence data, in the form of {@code JavaRDD<List<Writable>>}
     * and {@code JavaRDD<List<List<Writable>>>}.<br>
     * Note: this method performs a join by key. To perform this, we require that each record (and every step of
     * sequence records) contain the same key value (could be any Writable).<br>
     * <b>NOTE</b>: The RecordReaderMultiDataSetIterator <it>must</it> use {@link SparkSourceDummyReader} and
     * {@link SparkSourceDummySeqReader} instances in place of "real" RecordReader and SequenceRecordReader instances
     *
     * @param rdds      RDD with non-sequence data. May be null.
     * @param seqRdds   RDDs with sequence data. May be null.
     * @param rddsKeyColumns Column indices for the keys in the (non-sequence) RDDs data
     * @param seqRddsKeyColumns Column indices for the keys in the sequence RDDs data
     * @param filterMissing If true: filter out any records that don't have matching keys in all RDDs
     * @param iterator RecordReaderMultiDataSetIterator with {@link SparkSourceDummyReader} and {@link SparkSourceDummySeqReader}readers
     */
    public static JavaRDD<MultiDataSet> mapRRMDSI(List<JavaRDD<List<Writable>>> rdds, List<JavaRDD<List<List<Writable>>>> seqRdds,
                                                  int[] rddsKeyColumns, int[] seqRddsKeyColumns, boolean filterMissing,
                                                  RecordReaderMultiDataSetIterator iterator){
        checkIterator(iterator, (rdds == null ? 0 : rdds.size()), (seqRdds == null ? 0 : seqRdds.size()));
        assertNullOrSameLength(rdds, rddsKeyColumns, false);
        assertNullOrSameLength(seqRdds, seqRddsKeyColumns, true);
        if((rdds == null || rdds.isEmpty()) && (seqRdds == null || seqRdds.isEmpty()) ){
            throw new IllegalArgumentException();
        }

        JavaPairRDD<Writable,DataVecRecord> allPairs = null;
        if(rdds != null){
            for( int i=0; i<rdds.size(); i++ ){
                JavaRDD<List<Writable>> rdd = rdds.get(i);
                JavaPairRDD<Writable,DataVecRecord> currPairs = rdd.mapToPair(new MapToPairFn(i, rddsKeyColumns[i]));
                if(allPairs == null){
                    allPairs = currPairs;
                } else {
                    allPairs = allPairs.union(currPairs);
                }
            }
        }


        if(seqRdds != null){
            for( int i=0; i<seqRdds.size(); i++ ){
                JavaRDD<List<List<Writable>>> rdd = seqRdds.get(i);
                JavaPairRDD<Writable,DataVecRecord> currPairs = rdd.mapToPair(new MapToPairSeqFn(i, seqRddsKeyColumns[i]));
                if(allPairs == null){
                    allPairs = currPairs;
                } else {
                    allPairs = allPairs.union(currPairs);
                }
            }
        }

        int expNumRec = (rddsKeyColumns == null ? 0 : rddsKeyColumns.length);
        int expNumSeqRec = (seqRddsKeyColumns == null ? 0 : seqRddsKeyColumns.length);

        //Finally: group by key, filter (if necessary), convert
        JavaPairRDD<Writable, Iterable<DataVecRecord>> grouped = allPairs.groupByKey();
        if(filterMissing){
            //TODO
            grouped = grouped.filter(new FilterMissingFn(expNumRec, expNumSeqRec));
        }

        JavaRDD<DataVecRecords> combined = grouped.map(new CombineFunction(expNumRec, expNumSeqRec));
        return mapRRMDSIRecords(combined, iterator);
    }

    @AllArgsConstructor
    private static class MapToPairFn implements PairFunction<List<Writable>, Writable, DataVecRecord> {
        private int readerIdx;
        private int keyIndex;
        @Override
        public Tuple2<Writable, DataVecRecord> call(List<Writable> writables) throws Exception {
            return new Tuple2<>(writables.get(keyIndex), new DataVecRecord(readerIdx, writables, null));
        }
    }

    @AllArgsConstructor
    private static class MapToPairSeqFn implements PairFunction<List<List<Writable>>, Writable, DataVecRecord> {
        private int readerIdx;
        private int keyIndex;
        @Override
        public Tuple2<Writable, DataVecRecord> call(List<List<Writable>> seq) throws Exception {
            if(seq.isEmpty()){
                throw new IllegalStateException("Sequence of length 0 encountered");
            }
            return new Tuple2<>(seq.get(0).get(keyIndex), new DataVecRecord(readerIdx, null, seq));
        }
    }

    @AllArgsConstructor
    private static class CombineFunction implements Function<Tuple2<Writable, Iterable<DataVecRecord>>, DataVecRecords>{
        private int expNumRecords;
        private int expNumSeqRecords;
        @Override
        public DataVecRecords call(Tuple2<Writable, Iterable<DataVecRecord>> all) throws Exception {

            List<Writable>[] allRecordsArr = null;
            if(expNumRecords > 0){
                allRecordsArr = (List<Writable>[])new List[expNumRecords];  //Array.newInstance(List.class, expNumRecords);
            }
            List<List<Writable>>[] allRecordsSeqArr = null;
            if(expNumSeqRecords > 0){
                allRecordsSeqArr = (List<List<Writable>>[])new List[expNumSeqRecords];
            }

            for(DataVecRecord rec : all._2()){
                if(rec.getRecord() != null){
                    allRecordsArr[rec.getReaderIdx()] = rec.getRecord();
                } else {
                    allRecordsSeqArr[rec.getReaderIdx()] = rec.getSeqRecord();
                }
            }

            if(allRecordsArr != null){
                for(int i=0; i<allRecordsArr.length; i++ ){
                    if(allRecordsArr[i] == null){
                        throw new IllegalStateException("Encountered null records for input index " + i);
                    }
                }
            }

            if(allRecordsSeqArr != null){
                for(int i=0; i<allRecordsSeqArr.length; i++ ){
                    if(allRecordsSeqArr[i] == null){
                        throw new IllegalStateException("Encountered null sequence records for input index " + i);
                    }
                }
            }

            List<List<Writable>> r = (allRecordsArr == null ? null : Arrays.asList(allRecordsArr));
            List<List<List<Writable>>> sr = (allRecordsSeqArr == null ? null : Arrays.asList(allRecordsSeqArr));
            return new DataVecRecords(r, sr);
        }
    }


    @AllArgsConstructor
    private static class FilterMissingFn implements Function<Tuple2<Writable, Iterable<DataVecRecord>>, Boolean>{
        private final int expNumRec;
        private final int expNumSeqRec;
        private transient ThreadLocal<Set<Integer>> recIdxs;
        private transient ThreadLocal<Set<Integer>> seqRecIdxs;

        private FilterMissingFn(int expNumRec, int expNumSeqRec){
            this.expNumRec = expNumRec;
            this.expNumSeqRec = expNumSeqRec;
        }

        @Override
        public Boolean call(Tuple2<Writable, Iterable<DataVecRecord>> iter) throws Exception {
            if(recIdxs == null) recIdxs = new ThreadLocal<>();
            if(seqRecIdxs == null) seqRecIdxs = new ThreadLocal<>();

            Set<Integer> ri = recIdxs.get();
            if(ri == null){
                ri = new HashSet<>();
                recIdxs.set(ri);
            }
            Set<Integer> sri = seqRecIdxs.get();
            if(sri == null){
                sri = new HashSet<>();
                seqRecIdxs.set(sri);
            }

            for(DataVecRecord r : iter._2()){
                if(r.getRecord() != null){
                    ri.add(r.getReaderIdx());
                } else if(r.getSeqRecord() != null){
                    sri.add(r.getReaderIdx());
                }
            }

            int count = ri.size();
            int count2 = sri.size();

            ri.clear();
            sri.clear();

            return (count == expNumRec) && (count2 == expNumSeqRec);
        }
    }


    private static void assertNullOrSameLength(List<?> list, int[] arr, boolean isSeq){
        if(list != null && arr == null){
            throw new IllegalStateException();
        }
        if(list == null && (arr != null && arr.length > 0)){
            throw new IllegalStateException();
        }
        if(list != null && list.size() != arr.length){
            throw new IllegalStateException();
        }
    }


    public static JavaRDD<MultiDataSet> mapRRMDSIRecords(JavaRDD<DataVecRecords> rdd, RecordReaderMultiDataSetIterator iterator){
        return rdd.map(new RRMDSIFunction(iterator));
    }

    private static void checkIterator( RecordReaderMultiDataSetIterator iterator, int maxReaders, int maxSeqReaders ){


        Map<String,RecordReader> rrs = iterator.getRecordReaders();
        Map<String,SequenceRecordReader> seqRRs = iterator.getSequenceRecordReaders();


        if(rrs != null && rrs.size() > maxReaders){
            throw new IllegalStateException("Invalid state: iterator has " + rrs.size() + " readers but " + maxReaders
                    + " RDDs of List<Writable> were provided");
        }
        if(seqRRs != null && seqRRs.size() > maxSeqReaders){
            throw new IllegalStateException("Invalid state: iterator has " + seqRRs.size() + " sequence readers but " +
                    maxSeqReaders + " RDDs of sequences - List<List<Writable>> were provided");
        }

        if(rrs != null && rrs.size() > 0){
            for(Map.Entry<String,RecordReader> e : rrs.entrySet()){
                if(!(e.getValue() instanceof SparkSourceDummyReader)){
                    throw new IllegalStateException("Invalid state: expected SparkSourceDummyReader for reader with name \""
                            + e.getKey() + "\", but got reader type: " + e.getKey().getClass());
                }
            }
        }

        if(seqRRs != null && seqRRs.size() > 0){
            for(Map.Entry<String,SequenceRecordReader> e : seqRRs.entrySet()){
                if(!(e.getValue() instanceof SparkSourceDummySeqReader)){
                    throw new IllegalStateException("Invalid state: expected SparkSourceDummySeqReader for sequence reader with name \""
                            + e.getKey() + "\", but got reader type: " + e.getKey().getClass());
                }
            }
        }
    }
}
