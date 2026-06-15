open module nd4j.torchscript {
    requires nd4j.api;
    requires nd4j.common;
    requires slf4j.api;
    requires jackson;
    exports org.nd4j.torchscript;
    exports org.nd4j.torchscript.architecture;
    exports org.nd4j.torchscript.convert;
    exports org.nd4j.torchscript.format;
    exports org.nd4j.torchscript.ir;
}
