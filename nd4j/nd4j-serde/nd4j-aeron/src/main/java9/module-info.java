open module nd4j.aeron {
    requires guava;
    requires slf4j.api;
    requires io.aeron.all;
    requires nd4j.api;
    requires nd4j.common;
    exports org.nd4j.aeron.ipc;
    exports org.nd4j.aeron.ipc.chunk;
    exports org.nd4j.aeron.ipc.response;
    exports org.nd4j.aeron.ndarrayholder;
    exports org.nd4j.aeron.util;
}
