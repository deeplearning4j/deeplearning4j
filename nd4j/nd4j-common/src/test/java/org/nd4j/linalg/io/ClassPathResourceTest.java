package org.nd4j.linalg.io;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;

import static org.junit.Assert.assertEquals;

public class ClassPathResourceTest {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Test
    public void testDirExtractingIntelliJ() throws Exception {
        //https://github.com/deeplearning4j/deeplearning4j/issues/6483

        ClassPathResource cpr = new ClassPathResource("somedir");

        File f = testDir.newFolder();

        cpr.copyDirectory(f);

        File[] files = f.listFiles();
        assertEquals(1, files.length);
        assertEquals("afile.txt", files[0].getName());
    }

}
