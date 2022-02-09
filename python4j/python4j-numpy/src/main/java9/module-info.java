open module python4j.numpy {
    requires lombok;
    requires nd4j.cpu.api;
    requires org.bytedeco.cpython;
    requires org.bytedeco.javacpp;
    requires org.bytedeco.numpy;
    requires slf4j.api;
    requires nd4j.api;
    requires python4j.core;
    exports org.nd4j.python4j.numpy;
    provides org.nd4j.python4j.PythonType with org.nd4j.python4j.numpy.NumpyArray;
}
