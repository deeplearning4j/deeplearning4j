package org.nd4j.tensorflow.conversion;

import com.github.os72.protobuf351.util.JsonFormat;
import org.apache.commons.io.IOUtils;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.tensorflow.conversion.graphrunner.GraphRunner;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.bytedeco.javacpp.tensorflow.ConfigProto;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

public class GraphRunnerTest {

    @Test
    public void testGraphRunner() throws Exception {
        byte[] content = IOUtils.toByteArray(new ClassPathResource("/tf_graphs/nd4j_convert/simple_graph/frozen_model.pb").getInputStream());

        try(GraphRunner graphRunner = new GraphRunner(content)) {
            org.tensorflow.framework.ConfigProto.Builder builder = org.tensorflow.framework.ConfigProto.newBuilder();
            String json = graphRunner.sessionOptionsToJson();
            JsonFormat.parser().merge(json,builder);
            org.tensorflow.framework.ConfigProto build = builder.build();
            assertEquals(build,graphRunner.getProtoBufConfigProto());
            assertNotNull(graphRunner.getInputsForGraph());
            assertNotNull(graphRunner.getOutputsForGraph());


            org.tensorflow.framework.ConfigProto configProto1 = GraphRunner.fromJson(json);

            assertEquals(graphRunner.getProtoBufConfigProto(),configProto1);
            assertEquals(2,graphRunner.getInputsForGraph().size());
            assertEquals(1,graphRunner.getOutputsForGraph().size());

            INDArray input1 = Nd4j.linspace(1,4,4).reshape(4);
            INDArray input2 = Nd4j.linspace(1,4,4).reshape(4);

            Map<String,INDArray> inputs = new LinkedHashMap<>();
            inputs.put("input_0",input1);
            inputs.put("input_1",input2);

            for(int i = 0; i < 2; i++) {
                Map<String,INDArray> outputs = graphRunner.run(inputs);

                INDArray assertion = input1.add(input2);
                assertEquals(assertion,outputs.get("output"));
            }

        }
    }





}
