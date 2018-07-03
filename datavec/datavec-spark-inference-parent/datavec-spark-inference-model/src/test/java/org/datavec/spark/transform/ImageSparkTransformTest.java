package org.datavec.spark.transform;

import org.datavec.image.transform.ImageTransformProcess;
import org.datavec.spark.transform.model.Base64NDArrayBody;
import org.datavec.spark.transform.model.BatchImageRecord;
import org.datavec.spark.transform.model.SingleImageRecord;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.serde.base64.Nd4jBase64;

import java.io.File;

import static org.junit.Assert.assertEquals;

/**
 * Created by kepricon on 17. 5. 24.
 */
public class ImageSparkTransformTest {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Test
    public void testSingleImageSparkTransform() throws Exception {
        int seed = 12345;

        File f1 = new ClassPathResource("datavec-spark-inference/testimages/class1/A.jpg").getFile();

        SingleImageRecord imgRecord = new SingleImageRecord(f1.toURI());

        ImageTransformProcess imgTransformProcess = new ImageTransformProcess.Builder().seed(seed)
                        .scaleImageTransform(10).cropImageTransform(5).build();

        ImageSparkTransform imgSparkTransform = new ImageSparkTransform(imgTransformProcess);
        Base64NDArrayBody body = imgSparkTransform.toArray(imgRecord);

        INDArray fromBase64 = Nd4jBase64.fromBase64(body.getNdarray());
        System.out.println("Base 64ed array " + fromBase64);
        assertEquals(1, fromBase64.size(0));
    }

    @Test
    public void testBatchImageSparkTransform() throws Exception {
        int seed = 12345;

        File f0 = new ClassPathResource("datavec-spark-inference/testimages/class1/A.jpg").getFile();
        File f1 = new ClassPathResource("datavec-spark-inference/testimages/class1/B.png").getFile();
        File f2 = new ClassPathResource("datavec-spark-inference/testimages/class1/C.jpg").getFile();

        BatchImageRecord batch = new BatchImageRecord();
        batch.add(f0.toURI());
        batch.add(f1.toURI());
        batch.add(f2.toURI());

        ImageTransformProcess imgTransformProcess = new ImageTransformProcess.Builder().seed(seed)
                        .scaleImageTransform(10).cropImageTransform(5).build();

        ImageSparkTransform imgSparkTransform = new ImageSparkTransform(imgTransformProcess);
        Base64NDArrayBody body = imgSparkTransform.toArray(batch);

        INDArray fromBase64 = Nd4jBase64.fromBase64(body.getNdarray());
        System.out.println("Base 64ed array " + fromBase64);
        assertEquals(3, fromBase64.size(0));
    }
}
