package org.datavec.spark.transform;

import org.datavec.api.util.ClassPathResource;
import org.datavec.spark.transform.model.ImageRecord;
import org.junit.Test;

import java.io.File;

/**
 * Created by kepricon on 17. 5. 24.
 */
public class ImageRecordTest {

    @Test
    public void testImageRecord() throws Exception {
        File f0 = new ClassPathResource("/testimages/class0/0.jpg").getFile();
        File f1 = new ClassPathResource("/testimages/class1/A.jpg").getFile();

        ImageRecord imgRecord = new ImageRecord(f0.toURI());

        // need jackson test?
    }
}
