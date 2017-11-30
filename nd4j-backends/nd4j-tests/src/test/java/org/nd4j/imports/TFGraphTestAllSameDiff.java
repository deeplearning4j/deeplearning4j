package org.nd4j.imports;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.util.Collection;
import java.util.Map;

import static org.nd4j.imports.TFGraphTestAllHelper.*;

/**
 * Created by susaneraly on 11/29/17.
 */
@RunWith(Parameterized.class)
public class TFGraphTestAllSameDiff {
    private Map<String, INDArray> inputs;
    private Map<String, INDArray> predictions;
    private String modelName;
    private static final ExecuteWith executeWith = ExecuteWith.SAMEDIFF;

    @Parameterized.Parameters
    public static Collection<Object[]> data() throws IOException {
        return fetchTestParams(executeWith);
    }

    public TFGraphTestAllSameDiff(Map<String, INDArray> inputs, Map<String, INDArray> predictions, String modelName) throws IOException {
        this.inputs = inputs;
        this.predictions = predictions;
        this.modelName = modelName;
    }

    @Test
    public void test() throws Exception {
        Nd4j.create(1);
        testSingle(inputs, predictions, modelName, executeWith);
    }
}
