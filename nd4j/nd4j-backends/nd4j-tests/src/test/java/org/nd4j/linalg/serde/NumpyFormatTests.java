package org.nd4j.linalg.serde;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.codec.binary.Hex;
import org.apache.commons.io.FileUtils;
import org.junit.Ignore;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.util.Arrays;

import static org.junit.Assert.assertTrue;

@Slf4j
public class NumpyFormatTests extends BaseNd4jTest {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    public NumpyFormatTests(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testToNpyFormat() throws Exception {

        //File dir = new File("C:\\develop\\dl4j-test-resources\\src\\main\\resources\\numpy_arrays");

        val dir = testDir.newFolder();
        new ClassPathResource("numpy_arrays/").copyDirectory(dir);

        File[] files = dir.listFiles();
        int cnt = 0;

        for(File f : files){
            if(!f.getPath().endsWith(".npy")){
                log.warn("Skipping: {}", f);
                continue;
            }

            String path = f.getAbsolutePath();
            int lastDot = path.lastIndexOf('.');
            int lastUnderscore = path.lastIndexOf('_');
            String dtype = path.substring(lastUnderscore+1, lastDot);
            System.out.println(path + " : " + dtype);

            DataType dt = DataType.fromNumpy(dtype);
            //System.out.println(dt);

            INDArray arr = Nd4j.arange(12).castTo(dt).reshape(3,4);
            byte[] bytes = Nd4j.toNpyByteArray(arr);
            byte[] expected = FileUtils.readFileToByteArray(f);
/*
            log.info("E: {}", Arrays.toString(expected));
            for( int i=0; i<expected.length; i++ ){
                System.out.print((char)expected[i]);
            }

            System.out.println();System.out.println();

            log.info("A: {}", Arrays.toString(bytes));
            for( int i=0; i<bytes.length; i++ ){
                System.out.print((char)bytes[i]);
            }
            System.out.println();
*/

            assertArrayEquals("Failed with file [" + f.getName() + "]", expected, bytes);
            cnt++;
        }

        assertTrue(cnt > 0);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
