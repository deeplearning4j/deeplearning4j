open module nd4j.cpu {
    requires commons.math3;
    requires org.bytedeco.openblas;
    requires slf4j.api;
    requires flatbuffers.java;
    requires nd4j.api;
    requires nd4j.common;
    requires nd4j.cpu.api;
    requires nd4j.cpu.preset;
    requires org.bytedeco.javacpp;
    requires nd4j.cpu.backend.common;
    exports org.nd4j.linalg.cpu.nativecpu.backend;
    exports org.nd4j.linalg.cpu.nativecpu.bindings;
    provides org.nd4j.linalg.factory.Nd4jBackend with org.nd4j.linalg.cpu.nativecpu.backend.CpuBackend;
}
