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
    exports org.nd4j.linalg.cpu.nativecpu;
    exports org.nd4j.linalg.cpu.nativecpu.bindings;
    exports org.nd4j.linalg.cpu.nativecpu.blas;
    exports org.nd4j.linalg.cpu.nativecpu.buffer;
    exports org.nd4j.linalg.cpu.nativecpu.cache;
    exports org.nd4j.linalg.cpu.nativecpu.compression;
    exports org.nd4j.linalg.cpu.nativecpu.ops;
    exports org.nd4j.linalg.cpu.nativecpu.rng;
    exports org.nd4j.linalg.cpu.nativecpu.workspace;
    provides org.nd4j.linalg.compression.NDArrayCompressor with org.nd4j.linalg.cpu.nativecpu.compression.CpuThreshold;
    provides org.nd4j.linalg.factory.Nd4jBackend with org.nd4j.linalg.cpu.nativecpu.CpuBackend;
}
