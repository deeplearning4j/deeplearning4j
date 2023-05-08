package org.eclipse.deeplearning4j.modelimport.onnx;

import lombok.SneakyThrows;
import onnx.Onnx;
import org.nd4j.shade.protobuf.TextFormat;


public class OnnxTestUtils {

    public static Onnx.ModelProto loadFromString(String input) throws Exception {
        return TextFormat.parse(input, Onnx.ModelProto.class);
    }

}
