package org.deeplearning4j.spark.parameterserver.callbacks;

import org.apache.spark.input.PortableDataStream;
import org.nd4j.linalg.dataset.MultiDataSet;

/**
 * @author raver119@gmail.com
 */
public interface PortableDataStreamMDSCallback {

    /**
     * This method should do something, and return DataSet after all
     * @param pds
     * @return
     */
    MultiDataSet compute(PortableDataStream pds);
}
