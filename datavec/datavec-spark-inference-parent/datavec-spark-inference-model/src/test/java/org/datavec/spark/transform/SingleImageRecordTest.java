package org.datavec.spark.transform;

import org.datavec.spark.transform.model.SingleImageRecord;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;

/**
 * Created by kepricon on 17. 5. 24.
 */
public class SingleImageRecordTest {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Test
    public void testImageRecord() throws Exception {
        File f = testDir.newFolder();
        new ClassPathResource("datavec-spark-inference/testimages/").copyDirectory(f);
        File f0 = new File(f, "class0/0.jpg");
        File f1 = new File(f, "/class1/A.jpg");

        SingleImageRecord imgRecord = new SingleImageRecord(f0.toURI());

        // need jackson test?
    }
}
