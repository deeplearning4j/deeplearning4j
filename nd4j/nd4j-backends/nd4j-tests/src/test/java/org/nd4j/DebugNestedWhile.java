package org.nd4j;

import org.junit.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.graph.ui.LogFileWriter;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

public class DebugNestedWhile {

    @Test
    public void test() throws Exception {
        String path = "C:\\DL4J\\Git\\dl4j-test-resources\\src\\main\\resources\\tf_graphs\\examples\\simplewhile_nested";
        File f = new File(path, "frozen_model.pb");

        SameDiff sd = TFGraphMapper.getInstance().importGraph(f);

        File f2 = new File("C:/Temp/outputFile.bin");
        LogFileWriter lfw = new LogFileWriter(f2);
        lfw.writeGraphStructure(sd);
        lfw.writeFinishStaticMarker();

        INDArray in0 = Nd4j.linspace(1,4,4, DataType.FLOAT).reshape(2,2);
        INDArray in1 = Nd4j.create(DataType.FLOAT, new long[]{3,3}).assign(1.0);

        Map<String,INDArray> m = new HashMap<>();
        m.put("input_0", in0);
        m.put("input_1", in1);

        Map<String,INDArray> out = sd.exec(m, sd.outputs());

    }

}
