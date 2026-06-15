open module nd4j.ggml {
    requires nd4j.api;
    requires nd4j.common;
    requires slf4j.api;
    exports org.nd4j.ggml;
    exports org.nd4j.ggml.architecture;
    exports org.nd4j.ggml.convert;
    exports org.nd4j.ggml.export;
    exports org.nd4j.ggml.format;
    exports org.nd4j.ggml.quantization;
}
