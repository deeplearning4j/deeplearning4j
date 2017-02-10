package org.deeplearning4j.streaming.pipeline.spark.inerference;

import org.apache.commons.net.util.Base64;
import org.datavec.api.writable.Writable;
import org.datavec.spark.functions.FlatMapFunctionAdapter;
import org.datavec.spark.transform.BaseFlatMapFunctionAdaptee;
import org.deeplearning4j.streaming.conversion.ndarray.RecordToNDArray;
import org.deeplearning4j.streaming.serde.RecordDeSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import scala.Tuple2;

import java.util.Arrays;
import java.util.Collection;

/**
 * Flat maps a binary dataset string in to a
 * dataset
 * @author Adam Gibson
 */
public class NDArrayFlatMap extends BaseFlatMapFunctionAdaptee<Tuple2<String, String>, INDArray> {

    public NDArrayFlatMap(RecordToNDArray recordToDataSetFunction) {
        super(new NDArrayFlatMapAdapter(recordToDataSetFunction));
    }
}

/**
 * Flat maps a binary dataset string in to a
 * dataset
 * @author Adam Gibson
 */
class NDArrayFlatMapAdapter implements FlatMapFunctionAdapter<Tuple2<String, String>, INDArray> {
    private RecordToNDArray recordToDataSetFunction;

    public NDArrayFlatMapAdapter(RecordToNDArray recordToDataSetFunction) {
        this.recordToDataSetFunction = recordToDataSetFunction;
    }

    @Override
    public Iterable<INDArray> call(Tuple2<String, String> stringStringTuple2) throws Exception {
        try {
            byte[] bytes = Base64.decodeBase64(stringStringTuple2._2());
            Collection<Collection<Writable>> records = new RecordDeSerializer().deserialize("topic", bytes);
            INDArray d = recordToDataSetFunction.convert(records);
            return Arrays.asList(d);

        } catch (Exception e) {
            System.out.println("Error serializing");
        }



        return null;
    }
}
