package org.datavec.spark.transform.service;

import org.datavec.image.transform.ImageTransformProcess;
import org.datavec.spark.transform.model.Base64NDArrayBody;
import org.datavec.spark.transform.model.BatchImageRecord;
import org.datavec.spark.transform.model.ImageRecord;

import java.io.IOException;

/**
 * Created by kepricon on 17. 6. 19.
 */
public interface DataVecImageTransformService {

    void setTransformProcess(ImageTransformProcess transformProcess);

    ImageTransformProcess transformProcess();

    Base64NDArrayBody transformIncrementalArray(ImageRecord record) throws IOException;

    Base64NDArrayBody transformArray(BatchImageRecord batch);
}
