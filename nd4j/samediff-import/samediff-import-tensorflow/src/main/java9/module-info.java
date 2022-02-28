open module samediff.modelimport.tensorflow {
    requires commons.io;
    requires nd4j.common;
    requires kotlin.stdlib;
    requires nd4j.api;
    requires nd4j.tensorflow;
    requires protobuf;
    requires samediff.modelimport.api;
    exports org.nd4j.samediff.frameworkimport.tensorflow;
    exports org.nd4j.samediff.frameworkimport.tensorflow.context;
    exports org.nd4j.samediff.frameworkimport.tensorflow.definitions;
    exports org.nd4j.samediff.frameworkimport.tensorflow.importer;
    exports org.nd4j.samediff.frameworkimport.tensorflow.ir;
    exports org.nd4j.samediff.frameworkimport.tensorflow.opdefs;
    exports org.nd4j.samediff.frameworkimport.tensorflow.process;
    exports org.nd4j.samediff.frameworkimport.tensorflow.rule.attribute;
    exports org.nd4j.samediff.frameworkimport.tensorflow.rule.tensor;
    provides org.nd4j.samediff.frameworkimport.ImportGraphHolder with org.nd4j.samediff.frameworkimport.tensorflow.TensorflowImportGraph;
    provides org.nd4j.samediff.frameworkimport.opdefs.OpDescriptorLoader with org.nd4j.samediff.frameworkimport.tensorflow.opdefs.TensorflowOpDescriptorLoader;
}
