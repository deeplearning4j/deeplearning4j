open module nd4j.hexagon {
    requires nd4j.api;
    requires nd4j.common;
    requires nd4j.cpu.api;
    requires org.bytedeco.javacpp;
    requires org.slf4j;
    requires static lombok;
    exports org.nd4j.linalg.hexagon;
    exports org.nd4j.linalg.hexagon.ops;
}
