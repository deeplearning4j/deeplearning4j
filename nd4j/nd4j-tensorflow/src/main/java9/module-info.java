open module nd4j.tensorflow {
    requires commons.io;
    requires nd4j.common;
    requires protobuf;
    requires slf4j.api;
    requires nd4j.api;
    requires org.bytedeco.javacpp;
    requires org.bytedeco.tensorflow;
    exports org.nd4j.tensorflow.conversion;
    exports org.nd4j.tensorflow.conversion.graphrunner;
    provides org.nd4j.TFGraphRunnerService with org.nd4j.tensorflow.conversion.graphrunner.GraphRunnerServiceProvider;
}
