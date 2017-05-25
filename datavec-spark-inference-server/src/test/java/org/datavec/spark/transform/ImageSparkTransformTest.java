package org.datavec.spark.transform;

import org.datavec.api.util.ClassPathResource;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.ImageTransformProcess;
import org.datavec.image.transform.ScaleImageTransform;
import org.datavec.spark.transform.model.Base64NDArrayBody;
import org.datavec.spark.transform.model.ImageRecord;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.serde.base64.Nd4jBase64;

import java.io.File;
import java.util.Random;

/**
 * Created by kepricon on 17. 5. 24.
 */
public class ImageSparkTransformTest {

    @Test
    public void testImageSparkTransform() throws Exception {
        int seed = 12345;

        File f0 = new ClassPathResource("/testimages/class0/0.jpg").getFile();
        File f1 = new ClassPathResource("/testimages/class1/A.jpg").getFile();

//        ImageRecord imgRecord = new ImageRecord(f0.toURI());
        ImageRecord imgRecord = new ImageRecord(f1.toURI());

        ImageTransformProcess imgTransformProcess = new ImageTransformProcess.Builder()
                .seed(seed)
                .scaleImageTransform(10)
                .cropImageTransform(50)
                .build();

        ImageSparkTransform imgSparkTransform = new ImageSparkTransform(imgTransformProcess);
        Base64NDArrayBody body = imgSparkTransform.toArray(imgRecord);

        INDArray fromBase64 = Nd4jBase64.fromBase64(body.getNdarray());
        System.out.println("Base 64ed array " + fromBase64);
    }
}
