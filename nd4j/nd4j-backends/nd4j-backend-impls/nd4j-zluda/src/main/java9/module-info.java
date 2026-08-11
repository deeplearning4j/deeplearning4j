open module nd4j.zluda {
    requires nd4j.api;
    requires nd4j.cuda.backend.common;
    requires slf4j.api;
    exports org.nd4j.linalg.jzluda;
    provides org.nd4j.linalg.factory.Nd4jBackend with org.nd4j.linalg.jzluda.JZludaBackend;
}
