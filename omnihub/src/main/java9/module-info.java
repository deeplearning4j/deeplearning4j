open module omnihub {
    requires static lombok;
    requires java.compiler;
    requires resources;
    requires com.squareup.javapoet;
    requires samediff.pipeline.core;
    requires nd4j.api;
    requires deeplearning4j.nn;
    requires deeplearning4j.modelimport;
    requires samediff.modelimport.onnx;
    requires samediff.modelimport.tensorflow;
    requires kotlin.stdlib;
    requires jackson;
    requires org.apache.commons.io;
    requires org.apache.commons.lang3;
    requires org.slf4j;
    requires progressbar;
    requires com.google.gson;
    exports org.eclipse.deeplearning4j.omnihub;
    exports org.eclipse.deeplearning4j.omnihub.api;
    exports org.eclipse.deeplearning4j.omnihub.dsl;
    exports org.eclipse.deeplearning4j.omnihub.models;
}
