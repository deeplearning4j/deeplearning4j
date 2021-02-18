/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
package org.nd4j.tensorflow.conversion;

import junit.framework.TestCase;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.bytedeco.tensorflow.TF_Tensor;
import org.junit.Ignore;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.common.tests.BaseND4JTest;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.resources.Resources;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.protobuf.util.JsonFormat;
import org.nd4j.tensorflow.conversion.graphrunner.GraphRunner;
import org.nd4j.tensorflow.conversion.graphrunner.SavedModelConfig;
import org.tensorflow.framework.ConfigProto;
import org.tensorflow.framework.GPUOptions;

import java.io.File;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

public class GraphRunnerTest extends BaseND4JTest {

    @Override
    public DataType getDataType() {
        return DataType.FLOAT;
    }

    @Override
    public DataType getDefaultFPDataType() {
        return DataType.FLOAT;
    }

    public static ConfigProto getConfig(){
        String backend = Nd4j.getExecutioner().getEnvironmentInformation().getProperty("backend");
        if("CUDA".equalsIgnoreCase(backend)) {
            org.tensorflow.framework.ConfigProto configProto = org.tensorflow.framework.ConfigProto.getDefaultInstance();
            ConfigProto.Builder b = configProto.toBuilder().addDeviceFilters(TensorflowConversion.defaultDeviceForThread());
            return b.setGpuOptions(GPUOptions.newBuilder()
                    .setAllowGrowth(true)
                    .setPerProcessGpuMemoryFraction(0.5)
                    .build()).build();
        }
        return null;
    }

    @Test
    public void testGraphRunner() throws Exception {
        List<String> inputs = Arrays.asList("input_0","input_1");
        byte[] content = IOUtils.toByteArray(new ClassPathResource("/tf_graphs/nd4j_convert/simple_graph/frozen_model.pb").getInputStream());

        try(GraphRunner graphRunner = GraphRunner.builder().graphBytes(content).inputNames(inputs).sessionOptionsConfigProto(getConfig()).build()) {
            runGraphRunnerTest(graphRunner);
        }
    }

    @Test
    public void testGraphRunnerFilePath() throws Exception {
        List<String> inputs = Arrays.asList("input_0","input_1");
        byte[] content = FileUtils.readFileToByteArray(Resources.asFile("/tf_graphs/nd4j_convert/simple_graph/frozen_model.pb"));

        try(GraphRunner graphRunner = GraphRunner.builder().graphBytes(content).inputNames(inputs).sessionOptionsConfigProto(getConfig()).build()) {
            runGraphRunnerTest(graphRunner);
        }
    }

    @Test
    public void testInputOutputResolution() throws Exception {
        ClassPathResource lenetPb = new ClassPathResource("tf_graphs/lenet_frozen.pb");
        byte[] content = IOUtils.toByteArray(lenetPb.getInputStream());
        List<String> inputs = Arrays.asList("Reshape/tensor");
        try(GraphRunner graphRunner = GraphRunner.builder().graphBytes(content).inputNames(inputs).sessionOptionsConfigProto(getConfig()).build()) {
            assertEquals(1, graphRunner.getInputOrder().size());
            assertEquals(1, graphRunner.getOutputOrder().size());
        }
    }


    @Test @Ignore   //Ignored 2019/02/05: ssd_inception_v2_coco_2019_01_28 does not exist in test resources
    public void testMultiOutputGraph() throws Exception {
        List<String> inputs = Arrays.asList("image_tensor");
        byte[] content = IOUtils.toByteArray(new ClassPathResource("/tf_graphs/examples/ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb").getInputStream());
        try(GraphRunner graphRunner = GraphRunner.builder().graphBytes(content).inputNames(inputs).sessionOptionsConfigProto(getConfig()).build()) {
            String[] outputs = new String[]{"detection_boxes", "detection_scores", "detection_classes", "num_detections"};

            assertEquals(1, graphRunner.getInputOrder().size());
            System.out.println(graphRunner.getOutputOrder());
            assertEquals(4, graphRunner.getOutputOrder().size());
        }
    }

    private void runGraphRunnerTest(GraphRunner graphRunner) throws Exception {
        String json = graphRunner.sessionOptionsToJson();
        if( json != null ) {
            org.tensorflow.framework.ConfigProto.Builder builder = org.tensorflow.framework.ConfigProto.newBuilder();
            JsonFormat.parser().merge(json, builder);
            org.tensorflow.framework.ConfigProto build = builder.build();
            assertEquals(build,graphRunner.getSessionOptionsConfigProto());
        }
        assertNotNull(graphRunner.getInputOrder());
        assertNotNull(graphRunner.getOutputOrder());


        org.tensorflow.framework.ConfigProto configProto1 = json == null ? null : GraphRunner.fromJson(json);

        assertEquals(graphRunner.getSessionOptionsConfigProto(),configProto1);
        assertEquals(2,graphRunner.getInputOrder().size());
        assertEquals(1,graphRunner.getOutputOrder().size());

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


    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Test
    public void testGraphRunnerSavedModel() throws Exception {
        File f = testDir.newFolder("test");
        new ClassPathResource("/tf_saved_models/saved_model_counter/00000123/").copyDirectory(f);
        SavedModelConfig savedModelConfig = SavedModelConfig.builder()
                .savedModelPath(f.getAbsolutePath())
                .signatureKey("incr_counter_by")
                .modelTag("serve")
                .build();
        try(GraphRunner graphRunner = GraphRunner.builder().savedModelConfig(savedModelConfig).sessionOptionsConfigProto(getConfig()).build()) {
            INDArray delta = Nd4j.create(new float[] { 42 }, new long[0]);
            Map<String,INDArray> inputs = new LinkedHashMap<>();
            inputs.put("delta:0",delta);
            Map<String,INDArray> outputs = graphRunner.run(inputs);
            assertEquals(1, outputs.size());
            System.out.println(Arrays.toString(outputs.keySet().toArray(new String[0])));
            INDArray output = outputs.values().toArray(new INDArray[0])[0];
            assertEquals(42.0, output.getDouble(0), 0.0);
        }
    }

    @Test
    public void testGraphRunnerCast() {
        INDArray arr = Nd4j.linspace(1,4,4).castTo(DataType.FLOAT);
        TF_Tensor tensor = TensorflowConversion.getInstance().tensorFromNDArray(arr);
        TF_Tensor tf_tensor = GraphRunner.castTensor(tensor, TensorDataType.FLOAT,TensorDataType.DOUBLE);
        INDArray doubleNDArray = TensorflowConversion.getInstance().ndArrayFromTensor(tf_tensor);
        TestCase.assertEquals(DataType.DOUBLE,doubleNDArray.dataType());

        arr = arr.castTo(DataType.INT);
        tensor = TensorflowConversion.getInstance().tensorFromNDArray(arr);
        tf_tensor = GraphRunner.castTensor(tensor, TensorDataType.fromNd4jType(DataType.INT),TensorDataType.DOUBLE);
        doubleNDArray = TensorflowConversion.getInstance().ndArrayFromTensor(tf_tensor);
        TestCase.assertEquals(DataType.DOUBLE,doubleNDArray.dataType());

    }
}
