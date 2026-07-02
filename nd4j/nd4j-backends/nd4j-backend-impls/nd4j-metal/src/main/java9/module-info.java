open module nd4j.metal {
    requires nd4j.api;
    requires nd4j.cpu.api;
    requires org.bytedeco.javacpp;
    requires slf4j.api;
    exports org.nd4j.linalg.metal;
    exports org.nd4j.linalg.metal.ops;
}
