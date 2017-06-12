package org.datavec;

import lombok.AllArgsConstructor;
import org.datavec.api.transform.TransformProcess;
import org.datavec.spark.transform.model.Base64NDArrayBody;
import org.datavec.spark.transform.model.BatchRecord;
import org.datavec.spark.transform.model.CSVRecord;
import org.datavec.spark.transform.service.DataVecTransformService;

/**
 * Created by agibsonccc on 6/12/17.
 */
@AllArgsConstructor
public class DataVecTransformClient implements DataVecTransformService {
    private String url;

    /**
     * @param transformProcess
     */
    @Override
    public void setTransformProcess(TransformProcess transformProcess) {

    }

    /**
     * @return
     */
    @Override
    public TransformProcess transformProcess() {
        return null;
    }

    /**
     * @param transform
     * @return
     */
    @Override
    public CSVRecord transformIncremental(CSVRecord transform) {
        return null;
    }

    /**
     * @param batchRecord
     * @return
     */
    @Override
    public BatchRecord transform(BatchRecord batchRecord) {
        return null;
    }

    /**
     * @param batchRecord
     * @return
     */
    @Override
    public Base64NDArrayBody transformArray(BatchRecord batchRecord) {
        return null;
    }

    /**
     * @param csvRecord
     * @return
     */
    @Override
    public Base64NDArrayBody transformArrayIncremental(CSVRecord csvRecord) {
        return null;
    }
}
