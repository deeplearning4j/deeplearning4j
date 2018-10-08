/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.imports;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.autodiff.execution.NativeGraphExecutioner;
import org.nd4j.autodiff.execution.conf.ExecutionMode;
import org.nd4j.autodiff.execution.conf.ExecutorConfiguration;
import org.nd4j.autodiff.execution.conf.OutputMode;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.graph.FlatGraph;
import org.nd4j.graph.FlatNode;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.impl.controlflow.If;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.util.HashUtil;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.tensorflow.framework.GraphDef;

import java.io.DataInputStream;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.stream.Collectors;

import static org.junit.Assert.*;


@Slf4j
@Ignore
@RunWith(Parameterized.class)
public class TensorFlowImportTest extends BaseNd4jTest {
    private static ExecutorConfiguration configuration = ExecutorConfiguration.builder()
            .executionMode(ExecutionMode.SEQUENTIAL)
            .profilingMode(OpExecutioner.ProfilingMode.DISABLED)
            .gatherTimings(true)
            .outputMode(OutputMode.IMPLICIT)
            .build();

    public TensorFlowImportTest(Nd4jBackend backend) {
        super(backend);
    }


    @Override
    public char ordering() {
        return 'c';
    }

    @Before
    public void setUp() throws Exception {
    }

    @After
    public void tearDown() throws Exception {
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableDebugMode(false);
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableVerboseMode(false);
    }

    @Test
    public void testClassHolder() {
        DifferentialFunctionClassHolder.getInstance();
    }

    @Test
    public void testSingleExample_1() throws Exception{
        val g =TFGraphMapper.getInstance().importGraph(new File("C:\\Users\\raver\\Downloads\\mnist.pb"));

        val array = Nd4j.ones(1, 28, 28);
        g.associateArrayWithVariable(array, "flatten_1_input");

        //g.asFlatFile(new File("../../../libnd4j/tests_cpu/resources/mnist.fb"), ExecutorConfiguration.builder().outputMode(OutputMode.VARIABLE_SPACE).build());

        g.execAndEndResult();
    }


    @Test
    public void testAssertImport_1() throws Exception {
        val graph = TFGraphMapper.getInstance().importGraph(new File("C:\\Users\\raver\\Downloads\\test.pb"));
    }

    @Test
    public void testArgMaxImport_2() throws Exception {
        val graph = TFGraphMapper.getInstance().importGraph(new ClassPathResource("/tf_graphs/examples/reductions/argmax3,4,5_-1/frozen_graph.pbtxt").getInputStream());

        graph.asFlatFile(new File("../../../libnd4j/tests_cpu/resources/argmax_macos.fb"), ExecutorConfiguration.builder().outputMode(OutputMode.IMPLICIT).build());

        log.info(graph.asFlatPrint());
    }

    @Test
    public void testArgMaxImport_1() throws Exception {
        val graph = TFGraphMapper.getInstance().importGraph(new ClassPathResource("/tf_graphs/argmax.pb.txt").getInputStream());

        log.info(graph.asFlatPrint());
        val result = graph.execAndEndResult();

        val exp = Nd4j.trueVector(new double[]{2.0, 2.0, 2.0});

        assertEquals(exp, result);
    }


    @Test
    public void testIfStatementNodes() throws Exception {
        // /home/agibsonccc/code/dl4j-test-resources/src/main/resources/tf_graphs/examples/simple_cond/frozen_graph.pbtxt
        val resourceInputStream = new ClassPathResource("/tf_graphs/examples/simple_cond/frozen_model.pb").getInputStream();
        val mapper = TFGraphMapper.getInstance();
        val readGraph = TFGraphMapper.getInstance().parseGraphFrom(resourceInputStream);
        val nodes = mapper.nodesByName(readGraph);
        /**
         * Work backwards starting fom the condition id (usually a name containing condid/pred_id:

         */

        val firstInput = nodes.get("cond5/Merge");
        val ifNodes = mapper.nodesForIf(firstInput,readGraph);
        assertEquals(5,ifNodes.getFalseNodes().size());
        assertEquals(5,ifNodes.getTrueNodes().size());
        assertEquals(10,ifNodes.getCondNodes().size());


        val secondInput = nodes.get("cond6/Merge");
        val ifNodesTwo = mapper.nodesForIf(secondInput,readGraph);
        assertEquals(5,ifNodesTwo.getFalseNodes().size());
        assertEquals(5,ifNodesTwo.getTrueNodes().size());
        assertEquals(6,ifNodesTwo.getCondNodes().size());


        val parentContext = SameDiff.create();
        val ifStatement = new If();
        ifStatement.initFromTensorFlow(firstInput,parentContext,Collections.emptyMap(),readGraph);
        assertNotNull(ifStatement.getLoopBodyExecution());
        assertNotNull(ifStatement.getFalseBodyExecution());
        assertNotNull(ifStatement.getPredicateExecution());

    }

    @Test
    @Ignore
    public void testIfIgnoreWhileMerge() throws Exception {
        val resourceInputStream = new ClassPathResource("/tf_graphs/examples/simple_while/frozen_model.pb").getInputStream();
        val mapper = TFGraphMapper.getInstance();
        val readGraph = TFGraphMapper.getInstance().parseGraphFrom(resourceInputStream);
        val nodes = mapper.nodesByName(readGraph);
        val firstInput = nodes.get("output/Merge");
        assertNotNull(firstInput);
        assertFalse(mapper.isOpIgnoreException(firstInput));

        val resourceInputStreamIf = new ClassPathResource("/tf_graphs/examples/simple_cond/frozen_model.pb").getInputStream();
        val readGraphIf = TFGraphMapper.getInstance().parseGraphFrom(resourceInputStreamIf);
        val nodesif = mapper.nodesByName(readGraphIf);
        /**
         * Work backwards starting fom the condition id (usually a name containing condid/pred_id:

         */

        val secondInput = nodesif.get("cond5/Merge");
        assertNotNull(secondInput);
        assertTrue(mapper.isOpIgnoreException(secondInput));

    }


    @Test
    public void testHashEquality1() {
        long hash = HashUtil.getLongHash("Conv2D");
        assertEquals(-1637140380760460323L, hash);
    }


    @Test
    public void testHashEquality2() {
        long hash = HashUtil.getLongHash("switch");
        assertEquals(-1988317239813741487L, hash);
    }

    @Test
    public void testCustomOps1() {
        val map = Nd4j.getExecutioner().getCustomOperations();

        assertTrue(map.size() > 0);
    }

    @Test
    @Ignore
    public void importGraph1() throws Exception {
        SameDiff graph = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/max_add_2.pb.txt").getInputStream());

        assertNotNull(graph);

        assertEquals(2, graph.variableMap().size());

        SDVariable var0 = graph.variableMap().get("zeros");
        SDVariable var1 = graph.variableMap().get("ones");

        assertNotNull(var0);
        assertNotNull(var1);

        assertNotNull(var0.getArr());
        assertNotNull(var1.getArr());

        assertEquals(0.0, var0.getArr().sumNumber().doubleValue(), 1e-5);
        assertEquals(12.0, var1.getArr().sumNumber().doubleValue(), 1e-5);
    }


    @Test
    @Ignore
    public void importGraph2() throws Exception {
        SameDiff graph = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/tensorflow_inception_graph.pb").getInputStream());

        assertNotNull(graph);
    }


    @Test
    @Ignore
    public void importGraph3() throws Exception {
        SameDiff graph = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/max_log_reg.pb.txt").getInputStream());

        assertNotNull(graph);
    }

    @Test
    @Ignore
    public void testImportIris() throws Exception  {
        SameDiff graph = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/train_iris.pb").getInputStream());
        assertNotNull(graph);

    }


    @Test
    @Ignore
    public void importGraph4() throws Exception {
        SameDiff graph = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/max_multiply.pb.txt").getInputStream());

        assertNotNull(graph);

        val p0 = Nd4j.create(10, 10).assign(2.0);
        val p1 = Nd4j.create(10, 10).assign(3.0);

        graph.associateArrayWithVariable(p0,graph.variableMap().get("Placeholder"));
        graph.associateArrayWithVariable(p1, graph.variableMap().get("Placeholder_1"));


        graph.var("Placeholder", p0);
        graph.var("Placeholder_1", p1);

        val res = graph.execAndEndResult();



        assertEquals(6.0, res.meanNumber().doubleValue(), 1e-5);
    }



    @Test
    public void testLenet() throws Exception {
        /**
         * Produced with:
         * python  ~/anaconda2/lib/python2.7/site-packages/tensorflow/python/tools/freeze_graph.py  --input_graph=graph2.pb.txt  --input_checkpoint=test3.ckpt  --output_graph=graph_frozen2.pb  --output_node_name=output/BiasAdd --input_binary=False

         */

        Nd4j.create(1);
        val rawGraph = GraphDef.parseFrom(new ClassPathResource("tf_graphs/lenet_cnn.pb").getInputStream());
        val nodeNames = rawGraph.getNodeList().stream().map(node -> node.getName()).collect(Collectors.toList());
        System.out.println(nodeNames);
        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/lenet_cnn.pb").getInputStream());


        val convNode = tg.getVariable("conv2d/kernel");
        assertNotNull(convNode.getArr());
        val shape = convNode.getShape();
        System.out.println(Arrays.toString(shape));

        // this is NHWC weights. will be changed soon.
        assertArrayEquals(new int[]{5,5,1,32}, shape);
        System.out.println(convNode);
    }

    @Test
    public void testIntermediate2() throws Exception {
        Nd4j.create(1);
        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/max_lstm.pb").getInputStream());
    }

    @Test
    public void testIntermediate1() throws Exception {
        Nd4j.create(1);

        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/tensorflow_inception_graph.pb").getInputStream());

        assertTrue(tg.getVariable("input") != null);
        // assertTrue(tg.getVariableSpace().getVariable("input").isPlaceholder());

        val ipod = Nd4j.read(new DataInputStream(new ClassPathResource("tf_graphs/ipod.nd4").getInputStream()));

        tg.updateVariable("input",ipod);

        val buffer = tg.asFlatBuffers();
        assertNotNull(buffer);

    }



    @Test
    public void testIntermediateLoop1() throws Exception {
        Nd4j.create(1);
        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/simple_while.pb.txt").getInputStream());

        assertNotNull(tg);


        val graph = FlatGraph.getRootAsFlatGraph(tg.asFlatBuffers());

        assertEquals(6, graph.variablesLength());
//        assertEquals("alpha/Assign", graph.nodes(0).name());
    }


    @Test
    @Ignore
    public void testWeirdConvImport() {
        val tg = TFGraphMapper.getInstance().importGraph(new File("/home/agibsonccc/code/raver_tfimport_test1/profiling_conv.pb.txt"));
        assertNotNull(tg);
    }


    @Test
    public void testIntermediateLoop3() throws Exception {
        Nd4j.create(1);
        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/nested_while.pb.txt").getInputStream());

        assertNotNull(tg);

        // now converting to FlatBuffer
        val fb = tg.asFlatBuffers();
        assertNotNull(fb);

        val graph = FlatGraph.getRootAsFlatGraph(fb);
        assertEquals(15, graph.variablesLength());

        //assertEquals("phi/Assign", graph.nodes(0).name());
        //assertEquals("alpha/Assign", graph.nodes(1).name());

        assertEquals(2, graph.nodes(0).inputPairedLength());
        assertEquals(2, graph.nodes(1).inputPairedLength());

        //   tg.asFlatFile(new File("../../../libnd4j/tests_cpu/resources/nested_while.fb"));
    }



    @Test
    @Ignore
    public void testIntermediateStridedSlice1() throws Exception {
        Nd4j.create(1);
        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/tensor_slice.pb.txt").getInputStream());

        assertNotNull(tg);

        val constIn = tg.getVariable("StridedSlice/input");
        assertNotNull(constIn);

        val arr = tg.getArrForVarName(constIn.getVarName());
        assertEquals(139.5, arr.sumNumber().doubleValue(), 1e-5);


        // now converting to FlatBuffer
        val fb = tg.asFlatBuffers();
        assertNotNull(fb);

        val graph = FlatGraph.getRootAsFlatGraph(fb);
        assertEquals(5, graph.variablesLength());

        val nodeSlice = graph.nodes(0);

        assertEquals(14, nodeSlice.extraIntegerLength());

        val begin_mask = nodeSlice.extraInteger(0);
        val ellipsis_mask = nodeSlice.extraInteger(1);
        val end_mask = nodeSlice.extraInteger(2);
        val new_axis_mask = nodeSlice.extraInteger(3);
        val shrink_axis_mask = nodeSlice.extraInteger(4);

        assertEquals(0, begin_mask);
        assertEquals(0, ellipsis_mask);
        assertEquals(0, end_mask);
        assertEquals(0, new_axis_mask);
        assertEquals(0, shrink_axis_mask);

        val nodeSum = graph.nodes(1);

      /*  assertEquals("StridedSlice", nodeSlice.name());
        assertEquals("Sum", nodeSum.name());
*/
        assertEquals(4, nodeSlice.inputPairedLength());
        assertEquals(2, nodeSum.inputPairedLength());

        // we expect these inputs to be 5:0 and 6:0 respectively
        // where 5 (or 6) is a graph node id
        // and :0 is graph node output index, which is 0 because that's predefined variables
        // P.s. nodeSlice.id() should be equal to 5 :)
        val in0 = nodeSum.inputPaired(0);
        val in1 = nodeSum.inputPaired(1);
/*
        assertEquals(5, nodeSlice.id());
        assertEquals(7, nodeSum.id());

        assertEquals(nodeSlice.id(), in0.first());
        assertEquals(5, in0.first());

        assertEquals(6, in1.first());
        assertEquals(0, in1.second());
*/

//         tg.asFlatFile(new File("../../../libnd4j/tests_cpu/resources/tensor_slice.fb"), ExecutorConfiguration.builder().outputMode(OutputMode.IMPLICIT).build());
/*
        val executioner = new NativeGraphExecutioner();

        val exp = Nd4j.create(3, 1).assign(3);

        val results = executioner.executeGraph(tg, configuration);

        assertNotNull(results);
        assertEquals(1, results.length);
        assertEquals(73.5f, results[0].getFloat(0), 1e-5f);*/
    }

    @Test
    @Ignore
    public void testIntermediateTensorArraySimple1() throws Exception {
        Nd4j.create(1);
        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/tensor_array.pb.txt").getInputStream());
        tg.updateVariable("input_matrix",Nd4j.ones(3,2));

        assertNotNull(tg);

        val firstSlice = tg.getVariable("strided_slice");


        val fb = tg.asFlatBuffers();
        assertNotNull(fb);

        val graph = FlatGraph.getRootAsFlatGraph(fb);
        assertEquals(36, graph.variablesLength());

        assertTrue(graph.nodesLength() > 1);
     /*   assertEquals("strided_slice", graph.nodes(0).name());
        assertEquals("TensorArray", graph.nodes(1).name());
*/
        //   assertEquals(4, graph.nodes(0).inputPairedLength());

        //tg.asFlatFile(new File("../../../libnd4j/tests_cpu/resources/tensor_array.fb"));
    }

    @Test
    @Ignore
    public void testIntermediateTensorArrayLoop1() throws Exception {
        val input = Nd4j.linspace(1, 10, 10).reshape(5, 2);
        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/tensor_array_loop.pb.txt").getInputStream());
        tg.updateVariable("input_matrix",input);
        assertNotNull(tg);

        val fb = tg.asFlatBuffers();
        assertNotNull(fb);

        val graph = FlatGraph.getRootAsFlatGraph(fb);
        assertEquals(12, graph.variablesLength());

        val strided_slice = graph.nodes(0);

      /*  assertEquals("strided_slice", strided_slice.name());
        assertEquals("TensorArray", graph.nodes(1).name());
*/
        assertEquals(4, strided_slice.inputPairedLength());


        // we expect these inputs to be 1:0, 2:0, 3:0 and 4:0 respectively
        // where 1 (or 2/3/4) is a graph node id
        // and :0 is graph node output index, which is 0 because that's predefined variables
        val in0 = strided_slice.inputPaired(0);
        val in1 = strided_slice.inputPaired(1);
        val in2 = strided_slice.inputPaired(2);
        val in3 = strided_slice.inputPaired(3);

        assertEquals(2, in0.first());
        assertEquals(0, in0.second());

        assertEquals(3, in1.first());
        assertEquals(0, in1.second());

        assertEquals(4, in2.first());
        assertEquals(0, in2.second());

        assertEquals(5, in3.first());
        assertEquals(0, in3.second());
    }




    @Test
    public void testIntermediateReduction() throws Exception {
        Nd4j.create(1);
        SameDiff tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/reduce_dim.pb.txt").getInputStream());
        SDVariable sumResultVar = tg.getVariable("Sum");

      /*  val func = tg.getFunctionForVertexId(sumResultVar.getVertexId());
        assertEquals(0,func.getDimensions()[0]);
        assertEquals(3,tg.variables().size());
        assertNotNull(sumResultVar);
        assertNotNull(tg.getFunctionForVertexId(sumResultVar.getVertexId()));
        System.out.println(tg.variables());

        assertNotNull(func.getDimensions());
        assertEquals(0,func.getDimensions()[0]);*/

        ByteBuffer fb = tg.asFlatBuffers();
        assertNotNull(fb);

        FlatGraph graph = FlatGraph.getRootAsFlatGraph(fb);
        assertEquals(1, graph.nodesLength());
        assertEquals(2, graph.variablesLength());

        assertEquals("Sum", graph.nodes(0).name());

        FlatNode nodeSum = graph.nodes(0);
        assertEquals(2, nodeSum.inputPairedLength());


        // we expect these inputs to be 1:0 and 2:0 respectively
        // where 1 (or 2) is a graph node id
        // and :0 is graph node output index, which is 0 because that's predefined variables
        val in0 = nodeSum.inputPaired(0);
        val in1 = nodeSum.inputPaired(1);

        assertEquals(1, in0.first());
        assertEquals(0, in0.second());

        assertEquals(2, in1.first());
        assertEquals(0, in1.second());

        System.out.println(tg.summary());

        int dimensionsLength = nodeSum.dimensionsLength();
        assertEquals(1, dimensionsLength);
        int d = nodeSum.dimensions(0);
        assertEquals(1, d);


        //log.info("nodeSum inputs length: {}; inputPaired length: {}",nodeSum.inputLength(), nodeSum.inputPairedLength());

        //tg.asFlatFile(new File("../../../libnd4j/tests_cpu/resources/reduce_dim.fb"));
        val executioner = new NativeGraphExecutioner();

        val exp = Nd4j.create(3, 1).assign(3);

        val results = executioner.executeGraph(tg, configuration);

        assertNotNull(results);
        assertEquals(1, results.length);
        assertEquals(exp, results[0]);
    }

    @Test
    public void testDefaultArgs() {
        val op = Nd4j.getOpFactory().getOpByName("relu");

        val extras = op.extraArgs();
        assertTrue(extras.length == 1);
        val value = (Double) extras[0];

        assertEquals(0.0f, value.floatValue(), 1e-5f);
    }

    @Test
    public void testInferShape() throws IOException {
        /**
         * node {
         name: "input"
         op: "Placeholder"
         attr {
         key: "dtype"
         value {
         type: DT_FLOAT
         }
         }
         attr {
         key: "shape"
         value {
         shape {
         dim {
         size: -1
         }
         dim {
         size: 4
         }
         }
         }
         }
         }
         node {
         name: "bias"
         op: "Const"
         attr {
         key: "dtype"
         value {
         type: DT_FLOAT
         }
         }
         attr {
         key: "value"
         value {
         tensor {
         dtype: DT_FLOAT
         tensor_shape {
         dim {
         size: 4
         }
         }
         tensor_content: "\000\000\200?\000\000\000@\000\000@@\000\000\200@"
         }
         }
         }
         }
         node {
         name: "bias/read"
         op: "Identity"
         input: "bias"
         attr {
         key: "_class"
         value {
         list {
         s: "loc:@bias"
         }
         }
         }
         attr {
         key: "T"
         value {
         type: DT_FLOAT
         }
         }
         }
         node {
         name: "output"
         op: "BiasAdd"
         input: "input"
         input: "bias/read"
         attr {
         key: "data_format"
         value {
         s: "NHWC"
         }
         }
         attr {
         key: "T"
         value {
         type: DT_FLOAT
         }
         }
         }
         library {
         }

         */
        SameDiff graph = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/examples/bias_add/frozen_model.pb").getInputStream());
        assertNotNull(graph);

        INDArray input = Nd4j.linspace(1,40,40).reshape(10,4);
        INDArray expectedOutput = Nd4j.linspace(1,40,40).reshape(10,4).addRowVector(Nd4j.linspace(1,4,4));
        INDArray actual =  graph.execWithPlaceHolderAndEndResult(Collections.singletonMap("input",input));
        assertEquals(input,graph.getVariable("input").getArr());
        assertArrayEquals(input.shape(),graph.getShapeForVarName(graph.getVariable("input").getVarName()));
        assertEquals(expectedOutput,actual);
    }


    @Test
    public void testImportMapping1() throws Exception {
        Nd4j.create(1);
        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/examples/ae_00/frozen_model.pb").getInputStream());

        val variables = new HashMap<String, SDVariable>();
        for (val var : tg.variables()) {
            variables.put(var.getVarName(), var);
        }

        val functions = new HashMap<String, DifferentialFunction>();
        for (val func: tg.functions()) {
            val ownName = func.getOwnName();
            val outName = func.outputVariables()[0].getVarName();

            assertTrue("Missing ownName: [" + ownName +"]",variables.containsKey(ownName));
            assertEquals(ownName, outName);
        }
    }

    @Test
    public void testCondMapping1() throws Exception {
        Nd4j.create(1);
        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/examples/simpleif_0/frozen_model.pb").getInputStream());
        assertNotNull(tg);

        tg.asFlatFile(new File("../../../libnd4j/tests_cpu/resources/simpleif_0_1.fb"));
/*
        //log.info("{}", tg.asFlatPrint());
        val array = tg.execAndEndResult();
        val exp = Nd4j.create(2, 2).assign(-2);
        assertNotNull(array);
        assertEquals(exp, array);*/
    }

    @Test
    public void testCondMapping2() throws Exception {
        Nd4j.create(1);
        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/examples/simpleif_0/frozen_model.pb").getInputStream());
        assertNotNull(tg);

        val input = Nd4j.create(2, 2).assign(-1);
        tg.associateArrayWithVariable(input, tg.getVariable("input_0"));
        //tg.asFlatFile(new File("../../../libnd4j/tests_cpu/resources/simpleif_0.fb"));

        //log.info("{}", tg.asFlatPrint());
        val array = tg.execAndEndResult();
        val exp = Nd4j.create(2, 2).assign(1);
        assertNotNull(array);
        assertEquals(exp, array);
    }

    @Test
    public void testWhileMapping1() throws Exception {
        Nd4j.create(1);
        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/examples/simplewhile_0/frozen_model.pb").getInputStream());
        assertNotNull(tg);
        val input = Nd4j.create(2, 2).assign(1);
        tg.associateArrayWithVariable(input, tg.getVariable("input_0"));

        //tg.asFlatFile(new File("../../../libnd4j/tests_cpu/resources/simplewhile_0_3.fb"));

        //log.info("{}", tg.asFlatPrint());


        val array = tg.execAndEndResult();
        val exp = Nd4j.create(2, 2).assign(1);
        assertNotNull(array);
        assertEquals(exp, array);
    }

    @Test
    public void testWhileMapping2() throws Exception {
        Nd4j.create(1);
        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/examples/simplewhile_0/frozen_model.pb").getInputStream());
        assertNotNull(tg);
        val input = Nd4j.trueScalar(4.0);
        tg.associateArrayWithVariable(input, tg.getVariable("input_1"));

        tg.asFlatFile(new File("../../../libnd4j/tests_cpu/resources/simplewhile_0_4.fb"));

        //log.info("{}", tg.asFlatPrint());
/*
        val array = tg.execAndEndResult();
        val exp = Nd4j.create(2, 2).assign(2);
        assertNotNull(array);
        assertEquals(exp, array);*/
    }

    @Test
    public void testWhileMapping3() throws Exception {
        Nd4j.create(1);
        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/examples/simplewhile_0/frozen_model.pb").getInputStream());
        assertNotNull(tg);
        val input = Nd4j.trueScalar(9.0);
        tg.associateArrayWithVariable(input, tg.getVariable("input_1"));

        //tg.asFlatFile(new File("../../../libnd4j/tests_cpu/resources/simplewhile_0.fb"));

        //log.info("{}", tg.asFlatPrint());

        val array = tg.execAndEndResult();
        val exp = Nd4j.create(2, 2).assign(4);
        assertNotNull(array);
        assertEquals(exp, array);
    }


    @Test
    public void testWhileDualMapping1() throws Exception {
        Nd4j.create(1);
        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/examples/simplewhile_1/frozen_model.pb").getInputStream());
        assertNotNull(tg);
        val input0 = Nd4j.create(2, 2).assign(-4.0);
        val input1 = Nd4j.trueScalar(1.0);
        tg.associateArrayWithVariable(input0, tg.getVariable("input_0"));
        tg.associateArrayWithVariable(input1, tg.getVariable("input_1"));

        //tg.asFlatFile(new File("../../../libnd4j/tests_cpu/resources/simplewhile_1.fb"));

        //log.info("{}", tg.asFlatPrint());

        val array = tg.execAndEndResult();
        val exp = Nd4j.create(2, 2).assign(-1);
        assertNotNull(array);
        assertEquals(exp, array);
    }

    @Test
    public void testWhileDualMapping2() throws Exception {
        Nd4j.create(1);
        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/examples/simplewhile_1/frozen_model.pb").getInputStream());
        assertNotNull(tg);
        val input0 = Nd4j.create(2, 2).assign(-9.0);
        val input1 = Nd4j.trueScalar(1.0);
        tg.associateArrayWithVariable(input0, tg.getVariable("input_0"));
        tg.associateArrayWithVariable(input1, tg.getVariable("input_1"));

        //tg.asFlatFile(new File("../../../libnd4j/tests_cpu/resources/simplewhile_1.fb"));

        //log.info("{}", tg.asFlatPrint());

        val array = tg.execAndEndResult();
        val exp = Nd4j.create(2, 2).assign(-3);
        assertNotNull(array);
        assertEquals(exp, array);
    }


    @Test
    public void testMixedWhileCond1() throws Exception {
        Nd4j.create(1);
        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/examples/simplewhile_nested/frozen_model.pb").getInputStream());
        assertNotNull(tg);
        val input0 = Nd4j.create(2, 2).assign(1.0);
        val input1 = Nd4j.create(3, 3).assign(2.0);
        tg.associateArrayWithVariable(input0, tg.getVariable("input_0"));
        tg.associateArrayWithVariable(input1, tg.getVariable("input_1"));

        //tg.asFlatFile(new File("../../../libnd4j/tests_cpu/resources/simplewhile_nested.fb"));


        //log.info("{}", tg.asFlatPrint());

        val array = tg.execAndEndResult();
        //val array = tg.getVariable("output").getArr();
        val exp = Nd4j.create(2, 2).assign(15.0);
        assertNotNull(array);
        assertEquals(exp, array);
    }

    @Test
    @Ignore
    public void testProfConv() throws Exception {
        Nd4j.create(1);
        val tg = TFGraphMapper.getInstance().importGraph(new File("/home/raver119/develop/workspace/models/profiling_conv.pb.txt"));
        assertNotNull(tg);

        tg.asFlatFile(new File("../../../libnd4j/tests_cpu/resources/profiling_conv.fb"));
    }

    @Test
    @Ignore
    public void testCrash_119_matrix_diag() throws Exception {
        Nd4j.create(1);

        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/examples/partition_stitch_misc/frozen_model.pb").getInputStream());
        assertNotNull(tg);

        val input0 = Nd4j.create(2, 5, 4).assign(1.0);
        val input1 = Nd4j.create(2, 3, 5, 4).assign(2.0);
        val input2 = Nd4j.create(3, 1, 5, 4).assign(3.0);
        tg.associateArrayWithVariable(input0, tg.getVariable("input_0"));
        tg.associateArrayWithVariable(input1, tg.getVariable("input_1"));
        tg.associateArrayWithVariable(input2, tg.getVariable("input_2"));


        tg.asFlatFile(new File("../../../libnd4j/tests_cpu/resources/partition_stitch_misc.fb"));
    }

    @Test
    @Ignore
    public void testCrash_119_tensor_dot_misc() throws Exception {
        Nd4j.create(1);

        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/examples/tensor_dot_misc/frozen_model.pb").getInputStream());
        assertNotNull(tg);

        val input0 = Nd4j.create(36, 3, 4, 5).assign(1.0);
        val input1 = Nd4j.create(5, 5, 3, 4).assign(2.0);

        tg.associateArrayWithVariable(input0, tg.getVariable("input_a"));
        tg.associateArrayWithVariable(input1, tg.getVariable("input_b"));

        tg.asFlatFile(new File("../../../libnd4j/tests_cpu/resources/tensor_dot_misc.fb"));
    }

    @Test
    @Ignore
    public void testCrash_119_transpose() throws Exception {
        Nd4j.create(1);

        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/examples/transpose/frozen_model.pb").getInputStream());
        assertNotNull(tg);

        val input0 = Nd4j.create(new double[]{0.98114507, 0.96400015, 0.58669623, 0.60073098, 0.75425418, 0.44258752, 0.76373084, 0.96593234, 0.34067846}, new int[] {3, 3});
        val input1 = Nd4j.create(new double[]{0.98114507, 0.60073098, 0.76373084, 0.96400015, 0.75425418, 0.96593234, 0.58669623, 0.44258752, 0.34067846}, new int[] {3, 3});

        tg.associateArrayWithVariable(input0, tg.getVariable("input"));
        tg.associateArrayWithVariable(input1, tg.getVariable("input_1"));

        tg.asFlatFile(new File("../../../libnd4j/tests_cpu/resources/transpose.fb"));
    }

    @Test
    @Ignore
    public void testCrash_119_simpleif_0() throws Exception {
        Nd4j.create(1);

        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/examples/simpleif_0/frozen_model.pb").getInputStream());
        assertNotNull(tg);

        val input0 = Nd4j.create(new float[] {1, 2, 3, 4}, new int[] {2, 2});
        val input1 = Nd4j.trueScalar(11f);

        tg.associateArrayWithVariable(input0, tg.getVariable("input_0"));
        tg.associateArrayWithVariable(input1, tg.getVariable("input_1"));

        //tg.asFlatFile(new File("../../../libnd4j/tests_cpu/resources/simpleif_0.fb"));
    }

    @Test
    @Ignore
    public void testCrash_119_ae_00() throws Exception {
        Nd4j.create(1);

        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/examples/ae_00/frozen_model.pb").getInputStream());
        assertNotNull(tg);

        val input0 = Nd4j.create(new double[] {0.98174960, 0.44406342,  0.50100771,  1.00000000,  -0.94038386,  0.46501783, -0.49040590, 0.98153842, -0.00198260,  0.49108310, -0.06085236, 0.93523693, -0.05857396, -0.46633510, -0.02806635, -0.96879626, -0.03938015, -0.51578135, -0.06333921, -1.00000000}, new int[] {5, 4});

        tg.associateArrayWithVariable(input0, tg.getVariable("input"));

        tg.asFlatFile(new File("../../../libnd4j/tests_cpu/resources/ae_00.fb"));
    }

    @Test
    @Ignore
    public void testCrash_119_expand_dim() throws Exception {
        Nd4j.create(1);

        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/examples/expand_dim/frozen_model.pb").getInputStream());
        assertNotNull(tg);

        val input0 = Nd4j.create(new double[] {0.09753360, 0.76124972, 0.24693797, 0.13813169, 0.33144656, 0.08299957, 0.67197708, 0.80659380, 0.98274191, 0.63566073, 0.21592326, 0.54902743}, new int[] {3, 4});

        tg.associateArrayWithVariable(input0, tg.getVariable("input_0"));

        tg.asFlatFile(new File("../../../libnd4j/tests_cpu/resources/expand_dim.fb"));
    }

    @Test
    @Ignore
    public void testCrash_119_reduce_dim_false() throws Exception {
        Nd4j.create(1);

        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/reduce_dim.pb.txt").getInputStream());
        assertNotNull(tg);

        tg.asFlatFile(new File("../../../libnd4j/tests_cpu/resources/reduce_dim_false.fb"), ExecutorConfiguration.builder().outputMode(OutputMode.IMPLICIT).build());
    }

    @Test
    @Ignore
    public void testCrash_119_reduce_dim_true() throws Exception {
        Nd4j.create(1);

        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/reduce_dim_true.pb.txt").getInputStream());
        assertNotNull(tg);

        tg.asFlatFile(new File("../../../libnd4j/tests_cpu/resources/reduce_dim_true.fb"), ExecutorConfiguration.builder().outputMode(OutputMode.IMPLICIT).build());
    }

    @Test
    public void testTensorArray_119_1() throws Exception {
        Nd4j.create(1);

        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/tensor_array.pb.txt").getInputStream());
        assertNotNull(tg);

        val input_matrix = Nd4j.ones(3, 2);
        tg.associateArrayWithVariable(input_matrix, "input_matrix");

        val array = tg.execAndEndResult();

        val exp = Nd4j.create(new float[] {1, 1, 2, 2, 3, 3}, new int[]{3, 2});

        assertEquals(exp, array);
    }

    @Test
    public void testTensorArray_119_2() throws Exception {
        Nd4j.create(1);

        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/tensor_array_read.pb.txt").getInputStream());
        assertNotNull(tg);

        val input_matrix = Nd4j.ones(3, 2);
        tg.associateArrayWithVariable(input_matrix, "input_matrix");

        val array = tg.execAndEndResult();

        val exp = Nd4j.create(new float[] {2, 2}, new int[]{2});

        assertEquals(exp, array);
    }


    @Test
    public void testTensorArray_119_3() throws Exception {
        Nd4j.create(1);

        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/tensor_array_unstack.pb.txt").getInputStream());
        assertNotNull(tg);

        val array = tg.execAndEndResult();

        val exp = Nd4j.create(new float[] {5, 6, 7, 8}, new int[]{4});

        assertEquals(exp, array);
    }

    @Test
    public void testTensorArray_119_4() throws Exception {
        Nd4j.create(1);

        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/tensor_array_loop.pb.txt").getInputStream());
        assertNotNull(tg);


        val input_matrix = Nd4j.linspace(1, 10, 10).reshape(5, 2);
        tg.associateArrayWithVariable(input_matrix, "input_matrix");


        log.info("Graph: {}", tg.asFlatPrint());

        val array = tg.execAndEndResult();

        val exp = Nd4j.create(new float[] {3,6,  9,12,  15,18,  21,24,  27,30}, new int[]{5, 2});

        assertEquals(exp, array);
    }

    @Test(expected = ND4JIllegalStateException.class)
    public void testNonFrozenGraph1() throws Exception {
        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/examples/unfrozen_simple_ae.pb").getInputStream());
    }
}