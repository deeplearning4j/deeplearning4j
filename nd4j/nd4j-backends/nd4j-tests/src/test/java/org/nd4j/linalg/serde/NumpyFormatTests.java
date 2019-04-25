package org.nd4j.linalg.serde;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.codec.binary.Hex;
import org.apache.commons.io.FileUtils;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.io.File;
import java.util.Arrays;

@Slf4j
public class NumpyFormatTests extends BaseNd4jTest {

    public NumpyFormatTests(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    @Ignore
    public void testToNpyFormat() throws Exception {

        File dir = new File("C:\\develop\\dl4j-test-resources\\src\\main\\resources\\numpy_arrays");
        File[] files = dir.listFiles();

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
            System.out.println(dt);

            INDArray arr = Nd4j.arange(12).castTo(dt).reshape(3,4);
            byte[] bytes = Nd4j.toNpyByteArray(arr);
            byte[] expected = FileUtils.readFileToByteArray(f);

            log.info("E: {}", Arrays.toString(expected));
            for( int i=0; i<expected.length; i++ ){
//                System.out.printf("%c ", expected[i]);
                System.out.print((char)expected[i]);
            }
            System.out.println();System.out.println();
            log.info("A: {}", Arrays.toString(bytes));
            for( int i=0; i<bytes.length; i++ ){
//                System.out.printf("%c ", bytes[i]);
                System.out.print((char)bytes[i]);
            }
            System.out.println();


//            System.out.println();
//            System.out.println(Hex.encodeHex(bytes));
//            System.out.println(Hex.encodeHex(expected));
            assertArrayEquals("Failed with file [" + f.getName() + "]", expected, bytes);
            System.out.println("\n---------");
        }


    }

    @Override
    public char ordering() {
        return 'c';
    }
}
