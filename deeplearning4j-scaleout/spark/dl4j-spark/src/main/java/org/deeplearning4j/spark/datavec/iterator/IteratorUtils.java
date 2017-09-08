package org.deeplearning4j.spark.datavec.iterator;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.nd4j.linalg.dataset.api.MultiDataSet;

import java.util.Collections;
import java.util.List;
import java.util.Map;

public class IteratorUtils {


    public JavaRDD<MultiDataSet> mapRRMDSI(JavaRDD<List<Writable>> rdd, RecordReaderMultiDataSetIterator iterator){
        return mapRRMDIRecords(rdd.map(new Function<List<Writable>,DataVecRecords>(){
            @Override
            public DataVecRecords call(List<Writable> v1) throws Exception {
                return new DataVecRecords(null, Collections.singletonList(v1), null);
            }
        }), iterator);
    }


    public JavaRDD<MultiDataSet> mapRRMDIRecords(JavaRDD<DataVecRecords> rdd, RecordReaderMultiDataSetIterator iterator){
        checkIterator(iterator, 1, 0);
        return rdd.map(new RRMDSIFunction(iterator));
    }

    private static void checkIterator( RecordReaderMultiDataSetIterator iterator, int maxReaders, int maxSeqReaders ){


        Map<String,RecordReader> rrs = iterator.getRecordReaders();
        Map<String,SequenceRecordReader> seqRRs = iterator.getSequenceRecordReaders();


        if(rrs != null && rrs.size() > maxReaders){
            throw new IllegalStateException();
        }
        if(seqRRs != null && seqRRs.size() > maxSeqReaders){
            throw new IllegalStateException();
        }

        if(rrs != null && rrs.size() > 0){
            for(Map.Entry<String,RecordReader> e : rrs.entrySet()){
                if(!(e instanceof SparkSourceDummyReader)){
                    throw new IllegalStateException(e.getKey());
                }
            }
        }

        if(seqRRs != null && seqRRs.size() > 0){
            for(Map.Entry<String,SequenceRecordReader> e : seqRRs.entrySet()){
                if(!(e instanceof SparkSourceDummySeqReader)){
                    throw new IllegalStateException(e.getKey());
                }
            }
        }
    }
}
