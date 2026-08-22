open module nd4j.tpu {
    requires nd4j.api;
    requires nd4j.cpu.api;
    requires nd4j.cpu.backend.common;
    requires nd4j.tpu.preset;
    requires org.bytedeco.javacpp;
    requires org.bytedeco.openblas;
    requires slf4j.api;

    exports org.nd4j.linalg.jtpu;
    exports org.nd4j.linalg.jtpu.bindings;
    exports org.nd4j.linalg.jtpu.ops;

    provides org.nd4j.linalg.factory.Nd4jBackend
            with org.nd4j.linalg.jtpu.JTpuBackend;
}
