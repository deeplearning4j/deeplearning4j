open module nd4j.onnxruntime {
    requires nd4j.common;
    requires org.bytedeco.javacpp;
    requires nd4j.api;
    requires org.bytedeco.onnxruntime;
    requires slf4j.api;
    exports org.nd4j.onnxruntime.runner;
    exports org.nd4j.onnxruntime.util;
}
