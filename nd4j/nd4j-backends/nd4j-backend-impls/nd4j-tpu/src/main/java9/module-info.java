open module nd4j.tpu {
    requires nd4j.api;
    requires nd4j.cpu.api;
    requires org.bytedeco.javacpp;
    requires slf4j.api;
    exports org.nd4j.linalg.jtpu;
    exports org.nd4j.linalg.jtpu.ops;
}
