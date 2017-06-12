package org.datavec.spark.transform.service;

import org.datavec.api.transform.TransformProcess;
import org.datavec.spark.transform.model.Base64NDArrayBody;
import org.datavec.spark.transform.model.BatchRecord;
import org.datavec.spark.transform.model.CSVRecord;

/**
 * Created by agibsonccc on 6/12/17.
 */
public interface DataVecTransformService {

    /**
     *
     * @param transformProcess
     */
    void setTransformProcess(TransformProcess transformProcess);

    /**
     *
     * @return
     */
    TransformProcess transformProcess();

    /**
     *
     * @param transform
     * @return
     */
    CSVRecord transformIncremental(CSVRecord transform);

    /**
     *
     * @param batchRecord
     * @return
     */
    BatchRecord transform(BatchRecord batchRecord);

    /**
     *
     * @param batchRecord
     * @return
     */
    Base64NDArrayBody transformArray(BatchRecord batchRecord);

    /**
     *
     * @param csvRecord
     * @return
     */
    Base64NDArrayBody transformArrayIncremental(CSVRecord csvRecord);


}
