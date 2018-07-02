package org.nd4j.linalg.api.ndarray;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.ArrayUtils;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.checkutil.NDArrayCreationUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.primitives.Pair;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Iterator;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Created by susaneraly on 6/18/16.
 */
@Slf4j
@RunWith(Parameterized.class)
public class TestNdArrReadWriteTxt extends BaseNd4jTest {


    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    public TestNdArrReadWriteTxt(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void compareAfterWrite() throws Exception {
        int [] ranksToCheck = new int[] {0,1,2,3,4};
        for (int i=0; i<ranksToCheck.length;i++) {
            log.info("Checking read write arrays with rank " + ranksToCheck[i]);
            compareArrays(ranksToCheck[i],ordering(), testDir);
        }
    }

    public static void compareArrays(int rank, char ordering, TemporaryFolder testDir) throws Exception {
        List<Pair<INDArray, String>> all = NDArrayCreationUtil.getTestMatricesWithVaryingShapes(rank,ordering);
        Iterator<Pair<INDArray,String>> iter = all.iterator();
        int cnt = 0;
        while (iter.hasNext()) {
            File dir = testDir.newFolder();
            Pair<INDArray,String> currentPair = iter.next();
            INDArray origArray = currentPair.getFirst();
            //adding elements outside the bounds where print switches to scientific notation
            origArray.tensorAlongDimension(0,0).muli(0).addi(100000);
            origArray.putScalar(0,10001.1234);
            log.info("\nChecking shape ..." + currentPair.getSecond());
            //log.info("C:\n"+ origArray.dup('c').toString());
            log.info("F:\n"+ origArray.toString());
            Nd4j.writeTxt(origArray, new File(dir, "someArr.txt").getAbsolutePath());
            INDArray readBack = Nd4j.readTxt(new File(dir, "someArr.txt").getAbsolutePath());
            assertEquals("\nNot equal on shape " + ArrayUtils.toString(origArray.shape()), origArray, readBack);
            cnt++;
        }
    }

    @Override
    public char ordering() {
        return 'f';
    }
}
