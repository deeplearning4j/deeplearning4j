open module nd4j.tvm {
    requires nd4j.common;
    requires org.bytedeco.javacpp;
    requires slf4j.api;
    requires nd4j.api;
    requires org.bytedeco.tvm;
    exports org.nd4j.tvm.runner;
    exports org.nd4j.tvm.util;
}
