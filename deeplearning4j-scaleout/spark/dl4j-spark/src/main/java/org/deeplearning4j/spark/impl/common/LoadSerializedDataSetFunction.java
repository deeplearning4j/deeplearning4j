package org.deeplearning4j.spark.impl.common;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.input.PortableDataStream;
import org.nd4j.linalg.dataset.DataSet;

import java.io.InputStream;

/**
 * This is a function that is used to load a {@link DataSet} object using {@link DataSet#load(InputStream)}.
 *
 * @author Alex Black
 */
public class LoadSerializedDataSetFunction implements Function<PortableDataStream, DataSet> {
    @Override
    public DataSet call(PortableDataStream pds) throws Exception {
        try (InputStream is = pds.open()) {
            DataSet d = new DataSet();
            d.load(is);
            return d;
        }
    }
}
