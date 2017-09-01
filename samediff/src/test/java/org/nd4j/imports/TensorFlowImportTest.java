package org.nd4j.imports;

import com.google.protobuf.TextFormat;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.autodiff.opstate.OpExecAction;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SDGraph;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.impl.SDVariable;
import org.nd4j.linalg.io.ClassPathResource;

import java.util.List;

import static org.junit.Assert.*;

public class TensorFlowImportTest {
    @Before
    public void setUp() throws Exception {
    }

    @Test
    public void importGraph1() throws Exception {
        SameDiff graph = TensorFlowImport.importGraph(new ClassPathResource("tf_graphs/max_add_2.pb.txt").getFile());

        assertNotNull(graph);

        assertEquals(2, graph.getSameDiffVariables().size());
        assertEquals(2, graph.getGraph().getInputs().size());
        assertEquals(1, graph.getGraph().getOpOrder().getActions().size());

        List<OpExecAction> actions = graph.getGraph().getOpOrder().getActions();
        assertEquals(1, actions.size());

        OpState state = actions.get(0).getOpState();

        assertEquals(OpState.OpType.TRANSFORM, state.getOpType());
        assertEquals(0, state.getOpNum());

        SDVariable var0 = graph.getVariableMap().get("zeros");
        SDVariable var1 = graph.getVariableMap().get("ones");

        assertNotNull(var0);
        assertNotNull(var1);

        assertNotNull(var0.getArr());
        assertNotNull(var1.getArr());

        assertEquals(0.0, var0.getArr().sumNumber().doubleValue(), 1e-5);
        assertEquals(12.0, var1.getArr().sumNumber().doubleValue(), 1e-5);
    }


    @Test
    public void importGraph2() throws Exception {
        SameDiff graph = TensorFlowImport.importGraph(new ClassPathResource("tf_graphs/tensorflow_inception_graph.pb").getFile());

        assertNotNull(graph);
    }


    @Test
    public void importGraph3() throws Exception {
        SameDiff graph = TensorFlowImport.importGraph(new ClassPathResource("tf_graphs/max_log_reg.pb.txt").getFile());

        assertNotNull(graph);
    }
}