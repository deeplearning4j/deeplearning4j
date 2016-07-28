package org.deeplearning4j.streaming.pipeline.spark;

import org.apache.commons.net.util.Base64;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.streaming.conversion.dataset.RecordToDataSet;
import org.deeplearning4j.streaming.serde.RecordDeSerializer;
import org.nd4j.linalg.dataset.DataSet;
import scala.Tuple2;

import java.util.Arrays;
import java.util.Collection;

/**
 * Flat maps a binary dataset string in to a
 * dataset
 */
public class DataSetFlatmap implements FlatMapFunction<Tuple2<String, String>, DataSet> {
    private int numLabels;
    private RecordToDataSet recordToDataSetFunction;

    public DataSetFlatmap(int numLabels, RecordToDataSet recordToDataSetFunction) {
        this.numLabels = numLabels;
        this.recordToDataSetFunction = recordToDataSetFunction;
    }

    @Override
    public Iterable<DataSet> call(Tuple2<String, String> stringStringTuple2) throws Exception {
        try {
            byte[] bytes = Base64.decodeBase64(stringStringTuple2._2());
            Collection<Collection<Writable>> records = new RecordDeSerializer().deserialize("topic", bytes);
            DataSet d = recordToDataSetFunction.convert(records,numLabels);
            return Arrays.asList(d);

        } catch (Exception e) {
            System.out.println("Error serializing");
        }



        return null;
    }
}
