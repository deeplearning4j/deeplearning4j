package org.deeplearning4j.spark.parameterserver.callbacks;

import org.apache.spark.input.PortableDataStream;
import org.nd4j.linalg.dataset.MultiDataSet;

import java.io.DataInputStream;

/**
 * @author raver119@gmail.com
 */
public class MultiDataSetDeserializationCallback implements PortableDataStreamMDSCallback {

    @Override
    public MultiDataSet compute(PortableDataStream pds) {
        try (DataInputStream is = pds.open()) {
            // TODO: do something better here
            MultiDataSet ds = new MultiDataSet();
            ds.load(is);
            return ds;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
