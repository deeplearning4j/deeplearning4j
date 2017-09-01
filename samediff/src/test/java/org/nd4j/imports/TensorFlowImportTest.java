package org.nd4j.imports;

import com.google.protobuf.TextFormat;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.io.ClassPathResource;

import static org.junit.Assert.*;

public class TensorFlowImportTest {
    @Before
    public void setUp() throws Exception {
    }

    @Test
    public void importGraph1() throws Exception {
        SameDiff diff = TensorFlowImport.importGraph(new ClassPathResource("tf_graphs/max_add_2.pb.txt").getFile());


        assertNotNull(diff);
    }


    @Test
    public void importGraph2() throws Exception {
        SameDiff diff = TensorFlowImport.importGraph(new ClassPathResource("tf_graphs/tensorflow_inception_graph.pb").getFile());

        assertNotNull(diff);
    }

}