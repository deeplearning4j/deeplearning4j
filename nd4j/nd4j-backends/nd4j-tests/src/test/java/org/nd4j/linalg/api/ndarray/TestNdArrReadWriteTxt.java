package org.nd4j.linalg.api.ndarray;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.ArrayUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.checkutil.NDArrayCreationUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.primitives.Pair;

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

    public TestNdArrReadWriteTxt(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void compareAfterWrite() {
        int [] ranksToCheck = new int[] {0,1,2,3,4};
        for (int i=0; i<ranksToCheck.length;i++) {
            log.info("Checking read write arrays with rank " + ranksToCheck[i]);
            compareArrays(ranksToCheck[i],ordering());
        }
    }

    public static void compareArrays(int rank, char ordering) {
        List<Pair<INDArray, String>> all = NDArrayCreationUtil.getTestMatricesWithVaryingShapes(rank,ordering);
        Iterator<Pair<INDArray,String>> iter = all.iterator();
        while (iter.hasNext()) {
            Pair<INDArray,String> currentPair = iter.next();
            INDArray origArray = currentPair.getFirst();
            //adding elements outside the bounds where print switches to scientific notation
            origArray.tensorAlongDimension(0,0).muli(0).addi(100000);
            origArray.putScalar(0,10001.1234);
            log.info("\nChecking shape ..." + currentPair.getSecond());
            log.info("\n"+ origArray.dup('c').toString());
            Nd4j.writeTxt(origArray, "someArr.txt");
            INDArray readBack = Nd4j.readTxt("someArr.txt");
            assertEquals("\nNot equal on shape " + ArrayUtils.toString(origArray.shape()), origArray, readBack);
            try {
                Files.delete(Paths.get("someArr.txt"));
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    public char ordering() {
        return 'f';
    }
}
