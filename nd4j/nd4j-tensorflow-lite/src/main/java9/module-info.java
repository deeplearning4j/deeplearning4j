open module nd4j.tensorflow.lite {
    requires nd4j.common;
    requires org.bytedeco.javacpp;
    requires slf4j.api;
    requires nd4j.api;
    requires org.bytedeco.tensorflowlite;
    exports org.nd4j.tensorflowlite.runner;
    exports org.nd4j.tensorflowlite.util;
}
