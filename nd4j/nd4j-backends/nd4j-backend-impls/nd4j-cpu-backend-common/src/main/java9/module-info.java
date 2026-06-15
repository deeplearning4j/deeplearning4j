open module nd4j.cpu.backend.common {
    requires nd4j.api;
    requires nd4j.cpu.api;
    exports org.nd4j.linalg.cpu.nativecpu;
    exports org.nd4j.linalg.cpu.nativecpu.blas;
    exports org.nd4j.linalg.cpu.nativecpu.buffer;
    exports org.nd4j.linalg.cpu.nativecpu.cache;
    exports org.nd4j.linalg.cpu.nativecpu.compression;
    exports org.nd4j.linalg.cpu.nativecpu.ops;
    exports org.nd4j.linalg.cpu.nativecpu.rng;
    exports org.nd4j.linalg.cpu.nativecpu.workspace;
}
