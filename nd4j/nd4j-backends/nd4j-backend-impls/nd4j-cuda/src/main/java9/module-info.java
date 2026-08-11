open module nd4j.cuda {
    requires org.apache.commons.lang3;
    requires org.bytedeco.cuda;
    requires slf4j.api;
    requires flatbuffers.java;
    requires guava;
    requires nd4j.api;
    requires nd4j.common;
    requires nd4j.cpu.api;
    requires nd4j.cuda.preset;
    requires nd4j.cuda.backend.common;
    requires org.bytedeco.javacpp;
    exports org.nd4j.linalg.jcublas;
    provides org.nd4j.linalg.factory.Nd4jBackend with org.nd4j.linalg.jcublas.JCublasBackend;
}
