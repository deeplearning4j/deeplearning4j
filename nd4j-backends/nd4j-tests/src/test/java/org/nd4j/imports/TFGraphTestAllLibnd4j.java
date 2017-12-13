package org.nd4j.imports;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.util.*;

import static org.nd4j.imports.TFGraphTestAllHelper.checkOnlyOutput;
import static org.nd4j.imports.TFGraphTestAllHelper.fetchTestParams;

/**
 * Created by susaneraly on 11/29/17.
 */
@RunWith(Parameterized.class)
@Slf4j
public class TFGraphTestAllLibnd4j {
    private Map<String, INDArray> inputs;
    private Map<String, INDArray> predictions;
    private String modelName;
    private static final TFGraphTestAllHelper.ExecuteWith EXECUTE_WITH = TFGraphTestAllHelper.ExecuteWith.LIBND4J;
    private static final String[] SKIP_ARR = new String[] {
            "deep_mnist",
            "deep_mnist_no_dropout",
            "inception_v3_with_softmax"
    };
    public static final Set<String> SKIP_SET = new HashSet<>(Arrays.asList(SKIP_ARR));

    @Parameterized.Parameters
    public static Collection<Object[]> data() throws IOException {
        return fetchTestParams(EXECUTE_WITH);
    }

    public TFGraphTestAllLibnd4j(Map<String, INDArray> inputs, Map<String, INDArray> predictions, String modelName) throws IOException {
        this.inputs = inputs;
        this.predictions = predictions;
        this.modelName = modelName;
    }

    @Test
    public void test() throws Exception {
        Nd4j.create(1);
        if (SKIP_SET.contains(modelName)) {
            log.info("\n\tSKIPPED MODEL: " + modelName);
            return;
        }
        checkOnlyOutput(inputs, predictions, modelName, EXECUTE_WITH);
    }

}
