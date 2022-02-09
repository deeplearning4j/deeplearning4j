open module python4j.core {
    requires commons.io;
    requires nd4j.common;
    requires org.bytedeco.javacpp;
    requires slf4j.api;
    requires org.bytedeco.cpython;
    exports org.nd4j.python4j;
}
