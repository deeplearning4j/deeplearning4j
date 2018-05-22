package org.deeplearning4j.spark.parameterserver.callbacks;

import org.apache.spark.input.PortableDataStream;
import org.nd4j.linalg.dataset.DataSet;

import java.io.DataInputStream;

/**
 * @author raver119@gmail.com
 */
public class DataSetDeserializationCallback implements PortableDataStreamCallback {

    @Override
    public DataSet compute(PortableDataStream pds) {
        try (DataInputStream is = pds.open()) {
            // TODO: do something better here
            org.nd4j.linalg.dataset.DataSet ds = new org.nd4j.linalg.dataset.DataSet();
            ds.load(is);
            return ds;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
