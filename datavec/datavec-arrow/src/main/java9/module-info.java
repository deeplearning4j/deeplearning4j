open module datavec.arrow {
    requires java.nio;
    requires commons.io;
    requires slf4j.api;
    requires arrow.memory.core;
    requires arrow.vector;
    requires datavec.api;
    requires nd4j.api;
    requires nd4j.common;
    exports org.datavec.arrow;
    exports org.datavec.arrow.recordreader;
}
