package org.datavec.spark.transform;

import lombok.AllArgsConstructor;
import org.datavec.api.writable.Writable;
import org.datavec.common.RecordConverter;
import org.datavec.image.data.ImageWritable;
import org.datavec.image.transform.ImageTransformProcess;
import org.datavec.image.transform.ShowImageTransform;
import org.datavec.spark.transform.model.Base64NDArrayBody;
import org.datavec.spark.transform.model.BatchImageRecord;
import org.datavec.spark.transform.model.ImageRecord;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.serde.base64.Nd4jBase64;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by kepricon on 17. 5. 24.
 */
@AllArgsConstructor
public class ImageSparkTransform {
    private ImageTransformProcess imageTransformProcess;

    public Base64NDArrayBody toArray(ImageRecord record) throws IOException {
        ImageWritable record2 = imageTransformProcess.transformFileUriToInput(record.getUri());
        INDArray finalRecord = imageTransformProcess.executeArray(record2);

//        INDArray convert = RecordConverter.toArray(finalRecord);
        return new Base64NDArrayBody(Nd4jBase64.base64String(finalRecord));
    }

    public Base64NDArrayBody toArray(BatchImageRecord batch) throws IOException {
        List<List<Writable>> records = new ArrayList<>();
        for (ImageRecord imgRecord : batch.getRecords()) {
            ImageWritable record2 = imageTransformProcess.transformFileUriToInput(imgRecord.getUri());
            INDArray finalRecord = imageTransformProcess.executeArray(record2);
            List<Writable> writables = RecordConverter.toRecord(finalRecord);
            records.add(writables);
        }

        INDArray convert = RecordConverter.toMatrix(records);
        return new Base64NDArrayBody(Nd4jBase64.base64String(convert));
    }

}
