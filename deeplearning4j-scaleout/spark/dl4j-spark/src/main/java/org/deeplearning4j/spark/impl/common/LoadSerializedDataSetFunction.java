package org.deeplearning4j.spark.impl.common;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.input.PortableDataStream;
import org.nd4j.linalg.dataset.DataSet;
import scala.Tuple2;

import java.io.InputStream;

/**
 * Created by Alex on 02/07/2016.
 */
public class LoadSerializedDataSetFunction implements Function<Tuple2<String,PortableDataStream>,DataSet> {
    @Override
    public DataSet call(Tuple2<String,PortableDataStream> t2) throws Exception {
        try {
            InputStream is = t2._2().open();
            DataSet d = new DataSet();
            d.load(is);
            return d;
        } finally {
            t2._2().close();
        }
    }
}
