package org.nd4j.linalg.serde;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.io.*;

import static org.junit.Assert.assertEquals;

@RunWith(Parameterized.class)
@Slf4j
public class LargeSerDeTests extends BaseNd4jTest {
    public LargeSerDeTests(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testLargeArraySerDe_1() throws Exception {
        val arrayA = Nd4j.rand(new long[] {1, 135079944});
        //val arrayA = Nd4j.rand(new long[] {1, 13507});

        val tmpFile = File.createTempFile("sdsds", "sdsd");
        tmpFile.deleteOnExit();

        try (val fos = new FileOutputStream(tmpFile); val bos = new BufferedOutputStream(fos); val dos = new DataOutputStream(bos)) {
            Nd4j.write(arrayA, dos);
        }


        try (val fis = new FileInputStream(tmpFile); val bis = new BufferedInputStream(fis); val dis = new DataInputStream(bis)) {
            val arrayB = Nd4j.read(dis);

            assertArrayEquals(arrayA.shape(), arrayB.shape());
            assertEquals(arrayA.length(), arrayB.length());
            assertEquals(arrayA, arrayB);
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
