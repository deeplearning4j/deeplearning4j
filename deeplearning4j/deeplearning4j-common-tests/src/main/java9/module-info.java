open module deeplearning4j.common.tests {
    requires java.management;
    requires lombok;
    requires nd4j.common;
    requires org.bytedeco.javacpp;
    requires org.junit.jupiter.api;
    requires slf4j.api;
    requires nd4j.api;
    exports org.deeplearning4j;
}
