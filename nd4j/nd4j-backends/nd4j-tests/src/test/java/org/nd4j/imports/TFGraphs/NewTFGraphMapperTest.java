package org.nd4j.imports.TFGraphs;

import org.junit.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.util.Collections;
import java.util.Map;

public class NewTFGraphMapperTest {

    @Test
    public void test(){

        File f = new File("C:\\DL4J\\Git\\dl4j-test-resources\\src\\main\\resources\\tf_graphs\\examples\\arg_max\\rank2_dim1\\frozen_model.pb");

        SameDiff sd = TFGraphMapper.importGraph(f);

        System.out.println(sd.summary());

        Map<String, INDArray> m = sd.output(Collections.emptyMap(), "ArgMax");
        System.out.println(m.get("ArgMax"));

    }

    @Test
    public void test2(){

        File f = new File("C:\\DL4J\\Git\\dl4j-test-resources\\src\\main\\resources\\tf_graphs\\examples\\concat\\frozen_model.pb");

        SameDiff sd = TFGraphMapper.importGraph(f);

        System.out.println(sd.summary());

        Map<String, INDArray> m = sd.output(Collections.emptyMap(), "output");
        System.out.println(m.get("output"));

    }

}
