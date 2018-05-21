package org.datavec.spark.transform;

import org.datavec.spark.transform.model.SingleImageRecord;
import org.junit.Test;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;

/**
 * Created by kepricon on 17. 5. 24.
 */
public class SingleImageRecordTest {

    @Test
    public void testImageRecord() throws Exception {
        File f0 = new ClassPathResource("/testimages/class0/0.jpg").getFile();
        File f1 = new ClassPathResource("/testimages/class1/A.jpg").getFile();

        SingleImageRecord imgRecord = new SingleImageRecord(f0.toURI());

        // need jackson test?
    }
}
