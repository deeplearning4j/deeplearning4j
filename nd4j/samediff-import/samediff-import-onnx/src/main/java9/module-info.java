open module samediff.modelimport.onnx {
    requires commons.io;
    requires nd4j.common;
    requires org.apache.commons.lang3;
    requires org.bytedeco.javacpp;
    requires org.bytedeco.onnx;
    requires guava;
    requires kotlin.stdlib;
    requires nd4j.api;
    requires nd4j.onnxruntime;
    requires protobuf;
    requires samediff.modelimport.api;
    exports org.nd4j.samediff.frameworkimport.onnx;
    exports org.nd4j.samediff.frameworkimport.onnx.context;
    exports org.nd4j.samediff.frameworkimport.onnx.definitions;
    exports org.nd4j.samediff.frameworkimport.onnx.definitions.implementations;
    exports org.nd4j.samediff.frameworkimport.onnx.importer;
    exports org.nd4j.samediff.frameworkimport.onnx.ir;
    exports org.nd4j.samediff.frameworkimport.onnx.opdefs;
    exports org.nd4j.samediff.frameworkimport.onnx.optimize;
    exports org.nd4j.samediff.frameworkimport.onnx.process;
    exports org.nd4j.samediff.frameworkimport.onnx.rule.attribute;
    exports org.nd4j.samediff.frameworkimport.onnx.rule.tensor;
    provides org.nd4j.samediff.frameworkimport.ImportGraphHolder with org.nd4j.samediff.frameworkimport.onnx.OnnxImportGraph;
    provides org.nd4j.samediff.frameworkimport.opdefs.OpDescriptorLoader with org.nd4j.samediff.frameworkimport.onnx.opdefs.OnnxOpDescriptorLoader;
}
