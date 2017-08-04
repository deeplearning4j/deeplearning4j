package org.deeplearning4j.streaming.conversion.ndarray;

import org.datavec.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.Collection;

/**
 * A function convert from record to ndarrays
 * @author Adam Gibson
 */
public interface RecordToNDArray extends Serializable {

    /**
     * Converts a list of records in to 1 ndarray
     * @param records the records to convert
     * @return the collection of records
     * to convert
     */
    INDArray convert(Collection<Collection<Writable>> records);

}
