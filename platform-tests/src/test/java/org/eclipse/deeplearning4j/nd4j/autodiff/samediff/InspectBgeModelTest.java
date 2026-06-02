package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.util.Arrays;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

public class InspectBgeModelTest {

    @Test
    public void inspectBgeModelDtypes() throws Exception {
        File modelFile = new File(System.getProperty("user.home") +
                "/.kompile/models/bge-base-en-v1.5/bge-base-en-v1.5.sdz");
        assumeTrue(modelFile.exists(), "BGE model not found at " + modelFile);

        SameDiff sd = SameDiff.load(modelFile, false);

        System.out.println("\n=== DOUBLE variables in BGE model ===");
        int doubleCount = 0;
        int floatCount = 0;
        int otherCount = 0;
        for (SDVariable v : sd.variables()) {
            INDArray arr = v.getArr();
            if (arr == null) continue;
            DataType dt = arr.dataType();
            if (dt == DataType.DOUBLE) {
                doubleCount++;
                String valStr = arr.isScalar() ? " value=" + arr.getDouble(0) : "";
                System.out.println("  DOUBLE: " + v.name() +
                        " shape=" + Arrays.toString(arr.shape()) +
                        " rank=" + arr.rank() + valStr);
            } else if (dt == DataType.FLOAT) {
                floatCount++;
            } else {
                otherCount++;
            }
        }
        System.out.println("\nSummary: " + doubleCount + " DOUBLE, " +
                floatCount + " FLOAT, " + otherCount + " other");
        System.out.println("GraphExecutionMode: " + sd.getGraphExecutionMode());
    }
}
