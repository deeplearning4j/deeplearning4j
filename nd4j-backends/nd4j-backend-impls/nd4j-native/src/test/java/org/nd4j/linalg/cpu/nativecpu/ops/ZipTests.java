package org.nd4j.linalg.cpu.nativecpu.ops;

import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
@Ignore
public class ZipTests {

    @Test
    public void testZip() throws Exception {

        File testFile = File.createTempFile("adasda","Dsdasdea");

        INDArray arr = Nd4j.create(new double[]{1,2,3,4,5,6,7,8,9,0});

        final FileOutputStream fileOut = new FileOutputStream(testFile);
        final ZipOutputStream zipOut = new ZipOutputStream(fileOut);
        zipOut.putNextEntry(new ZipEntry("params"));
        Nd4j.write(zipOut, arr);
        zipOut.flush();
        zipOut.close();


        final FileInputStream fileIn = new FileInputStream(testFile);
        final ZipInputStream zipIn = new ZipInputStream(fileIn);
        ZipEntry entry = zipIn.getNextEntry();
        INDArray read = Nd4j.read(zipIn);
        zipIn.close();


        assertEquals(arr, read);
    }
}
