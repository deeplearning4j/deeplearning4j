open module nd4j.zluda {
    requires nd4j.api;
    requires nd4j.cuda;
    requires org.bytedeco.javacpp;
    requires slf4j.api;
    exports org.nd4j.linalg.jzluda;
}
