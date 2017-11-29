package org.nd4j.imports;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.autodiff.execution.NativeGraphExecutioner;
import org.nd4j.autodiff.execution.conf.ExecutionMode;
import org.nd4j.autodiff.execution.conf.ExecutorConfiguration;
import org.nd4j.autodiff.execution.conf.OutputMode;
import org.nd4j.autodiff.opstate.OpExecAction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.graph.FlatGraph;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.util.HashUtil;
import org.tensorflow.framework.GraphDef;

import java.io.DataInputStream;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import static org.junit.Assert.*;


@Slf4j
public class TensorFlowImportTest {
    private static ExecutorConfiguration configuration = ExecutorConfiguration.builder()
            .executionMode(ExecutionMode.SEQUENTIAL)
            .profilingMode(OpExecutioner.ProfilingMode.DISABLED)
            .gatherTimings(true)
            .outputMode(OutputMode.IMPLICIT)
            .build();

    @Before
    public void setUp() throws Exception {
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
        assertEquals(2, graph.getGraph().getInputs().size());
        assertEquals(1, graph.getGraph().getOpOrder().getActions().size());

        List<OpExecAction> actions = graph.getGraph().getOpOrder().getActions();
        assertEquals(1, actions.size());

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
    public void testImportIris() throws Exception  {
        SameDiff graph = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/train_iris.pb").getInputStream());
        assertTrue(graph.graph().numVertices() > 0);
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
        assertArrayEquals(new int[]{32,1,5,5},shape);
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
        assertEquals("alpha/Assign", graph.nodes(0).name());
    }
/*
    @Test
    public void testIntermediateLoop2() throws Exception {
        Nd4j.create(1);
        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/three_arg_while.pb.txt").getInputStream());

        val phi = tg.getVariable("phi");
        assertNotNull(phi);
        assertArrayEquals(new int[] {2, 2}, phi.getShape());

        //was 9
        val scopeCondition = tg.getFunction("");
        //was 10
        val scopeBody = tg.getFunction("");

        val whileNode = tg.getNode(11);
        assertEquals("while", whileNode.getOpName());

        assertNotNull(scopeCondition);
        assertNotNull(scopeBody);

        // checking condition ops first
        assertEquals(2, scopeCondition.size());
        val firstScopedNode = scopeCondition.getNodes().get(0);
        val secondScopedNode = scopeCondition.getNodes().get(1);

        val condConstA = tg.getVariableSpace().getVariable("while/Const");
        val condConstB = tg.getVariableSpace().getVariable("while/Less/y");


        val var5 = tg.getVariableSpace().getVariable(-5);
        val varC = tg.getVariableSpace().getVariable("Const_2");

        assertTrue(var5 == varC);

        val var6 = tg.getVariableSpace().getVariable(-6);
        assertEquals("omega", var6.getName());



        assertEquals("Sum", firstScopedNode.getOpName());
        assertEquals(1, firstScopedNode.getInputs().size());
        assertEquals(TIndex.makeOf(whileNode.getId()), firstScopedNode.getInputs().get(0));
        assertArrayEquals(new int[] {0, 1}, firstScopedNode.getOpState().getAxes());
//        assertEquals(condConstA.getId(), firstScopedNode.getInputs().get(1).getNode());

        assertEquals("Less", secondScopedNode.getOpName());
        assertEquals(2, secondScopedNode.getInputs().size());
        assertEquals(firstScopedNode.getId(), secondScopedNode.getInputs().get(0).getNode());
        assertEquals(condConstB.getId(), secondScopedNode.getInputs().get(1).getNode());

        // TODO: we probably want to get rid of identity step? or, let it be?
        assertEquals(6, scopeBody.size());

        val loopConstA = tg.getVariableSpace().getVariable("while/add/y");
        val loopConstB = tg.getVariableSpace().getVariable("while/add_1/y");

        val identity0 = scopeBody.getNode("while/Identity");
        val identity1 = scopeBody.getNode("while/Identity_1");
        val identity2 = scopeBody.getNode("while/Identity_2");
        val returnScope = scopeBody.lastNode();

        assertNotNull(identity0);
        assertNotNull(identity1);
        assertNotNull(identity2);
        assertNotNull(returnScope);

        // now we're validating Identity input, it's derived from While op
        assertEquals(TIndex.makeOf(whileNode.getId(), 0), identity0.getInputs().get(0));
        assertEquals(TIndex.makeOf(whileNode.getId(), 1), identity1.getInputs().get(0));
        assertEquals(TIndex.makeOf(whileNode.getId(), 2), identity2.getInputs().get(0));

        assertEquals(3, returnScope.getInputs().size());


        val bodyNode4 = scopeBody.getNodes().get(3);
        val bodyNode5 = scopeBody.getNodes().get(4);

        assertEquals(2, bodyNode4.getInputs().size());
        assertEquals(identity0.getId(), bodyNode4.getInputs().get(0).getNode());
        assertEquals(loopConstA.getId(), bodyNode4.getInputs().get(1).getNode());

        assertEquals(identity1.getId(), bodyNode5.getInputs().get(0).getNode());
        assertEquals(loopConstB.getId(), bodyNode5.getInputs().get(1).getNode());


        // Now, we're checking ops that will be executed after the cycle
        val constAddY0 = tg.getVariableSpace().getVariable("add/y");
        val constAddY1 = tg.getVariableSpace().getVariable("add_1/y");

        val nodeAdd0 = tg.getNode("add");
        val nodeAdd1 = tg.getNode("add_1");

        assertNotNull(nodeAdd0);
        assertNotNull(nodeAdd1);

        assertNotNull(constAddY0);
        assertNotNull(constAddY1);

        assertEquals(constAddY0.getId(), nodeAdd0.getInputs().get(1).getNode());
        assertEquals(TIndex.makeOf(whileNode.getId(), 0), nodeAdd0.getInputs().get(0));


        assertEquals(constAddY1.getId(), nodeAdd1.getInputs().get(1).getNode());
        assertEquals(TIndex.makeOf(whileNode.getId(), 1), nodeAdd1.getInputs().get(0));


        // now converting to FlatBuffer
        val fb = tg.asFlatBuffers();
        assertNotNull(fb);

        val offset = fb.position();

        log.info("Length: {}; Offset: {};", fb.capacity(), offset);
        val array = fb.array();

        try (val fos = new FileOutputStream("../../../libnd4j/tests_cpu/resources/three_args_while.fb"); val dos = new DataOutputStream(fos)) {
            dos.write(array, offset, array.length - offset);
        }

    }*/

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

        assertEquals("phi/Assign", graph.nodes(0).name());
        assertEquals("alpha/Assign", graph.nodes(1).name());

        assertEquals(2, graph.nodes(0).inputPairedLength());
        assertEquals(2, graph.nodes(1).inputPairedLength());

     //   tg.asFlatFile(new File("../../../libnd4j/tests_cpu/resources/nested_while.fb"));
    }

    @Test
    public void testIntermediateStridedSlice1() throws Exception {
        Nd4j.create(1);
        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/tensor_slice.pb.txt").getInputStream());

        assertNotNull(tg);

        val constIn = tg.getVariable("StridedSlice/input");
        assertNotNull(constIn);

        val arr = tg.getArrForVertexId(constIn.getVertexId());
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

        assertEquals("StridedSlice", nodeSlice.name());
        assertEquals("Sum", nodeSum.name());

        assertEquals(4, nodeSlice.inputPairedLength());
        assertEquals(2, nodeSum.inputPairedLength());

        // we expect these inputs to be 5:0 and 6:0 respectively
        // where 5 (or 6) is a graph node id
        // and :0 is graph node output index, which is 0 because that's predefined variables
        // P.s. nodeSlice.id() should be equal to 5 :)
        val in0 = nodeSum.inputPaired(0);
        val in1 = nodeSum.inputPaired(1);

        assertEquals(5, nodeSlice.id());
        assertEquals(7, nodeSum.id());

        assertEquals(nodeSlice.id(), in0.first());
        assertEquals(5, in0.first());

        assertEquals(6, in1.first());
        assertEquals(0, in1.second());


        // tg.asFlatFile(new File("../../../libnd4j/tests_cpu/resources/tensor_slice.fb"));

        val executioner = new NativeGraphExecutioner();

        val exp = Nd4j.create(3, 1).assign(3);

        val results = executioner.executeGraph(tg, configuration);

        assertNotNull(results);
        assertEquals(1, results.length);
        assertEquals(73.5f, results[0].getFloat(0), 1e-5f);
    }

    @Test
    public void testIntermediateTensorArraySimple1() throws Exception {
        Nd4j.create(1);
        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/tensor_array.pb.txt").getInputStream());
        tg.updateVariable("input_matrix",Nd4j.ones(3,2));

        assertNotNull(tg);

        val firstSlice = tg.getVariable("strided_slice");


        val fb = tg.asFlatBuffers();
        assertNotNull(fb);

        val graph = FlatGraph.getRootAsFlatGraph(fb);
        assertEquals(22, graph.variablesLength());

        assertEquals("strided_slice", graph.nodes(0).name());
        assertEquals("TensorArray", graph.nodes(1).name());

        assertEquals(4, graph.nodes(0).inputPairedLength());

        //tg.asFlatFile(new File("../../../libnd4j/tests_cpu/resources/tensor_array.fb"));
    }

    @Test
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

        assertEquals("strided_slice", strided_slice.name());
        assertEquals("TensorArray", graph.nodes(1).name());

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
        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/reduce_dim.pb.txt").getInputStream());
        val sumResultVar = tg.getVariable("Sum");
        val func = tg.getFunctionForVertexId(sumResultVar.getVertexId());
        assertEquals(0,func.getDimensions()[0]);
        assertEquals(3,tg.variables().size());
        assertNotNull(sumResultVar);
        assertNotNull(tg.getFunctionForVertexId(sumResultVar.getVertexId()));
        System.out.println(tg.variables());

        assertNotNull(func.getDimensions());
        assertEquals(0,func.getDimensions()[0]);

        val fb = tg.asFlatBuffers();
        assertNotNull(fb);

        val graph = FlatGraph.getRootAsFlatGraph(fb);
        assertEquals(1, graph.nodesLength());
        assertEquals(2, graph.variablesLength());

        assertEquals("Sum", graph.nodes(0).name());

        val nodeSum = graph.nodes(0);
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


        assertEquals(1, nodeSum.dimensions(1));


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
        assertArrayEquals(input.shape(),graph.getShapeForVertexId(graph.getVariable("input").getVertexId()));
        assertEquals(expectedOutput,actual);
    }
}