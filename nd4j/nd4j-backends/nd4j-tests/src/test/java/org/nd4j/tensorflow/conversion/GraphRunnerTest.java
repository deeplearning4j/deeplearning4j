package org.nd4j.tensorflow.conversion;

import org.apache.commons.io.IOUtils;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.tensorflow.conversion.graphrunner.GraphRunner;
import org.tensorflow.framework.GraphDef;

import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

public class GraphRunnerTest {

    @Test
    public void testGraphRunner() throws Exception {
        byte[] content = IOUtils.toByteArray(new ClassPathResource("/tf_graphs/nd4j_convert/simple_graph/frozen_model.pb").getInputStream());
        try(GraphRunner graphRunner = new GraphRunner(content)) {
            INDArray input1 = Nd4j.linspace(1,4,4);
            INDArray input2 = Nd4j.linspace(1,4,4);
            INDArray result = Nd4j.createUninitialized(input1.shape());
            Map<String,INDArray> inputs = new LinkedHashMap<>();
            inputs.put("input_0",input1);
            inputs.put("input_1",input2);

            Map<String,INDArray> outputs = new HashMap<>();
            outputs.put("output",result);
            GraphDef graphDef1 = GraphDef.parseFrom(content);
            for(int i = 0; i < graphDef1.getNodeCount(); i++) {

            }
            graphRunner.run(inputs, Arrays.asList("input_0","input_1"),outputs,Arrays.asList("output"));

        }
    }

}
