open module nd4j.parameter.server {
    requires guava;
    requires jcommander;
    requires json;
    requires nd4j.common;
    requires slf4j.api;
    requires unirest.java;
    requires io.aeron.all;
    requires jackson;
    requires nd4j.aeron;
    requires nd4j.api;
    requires nd4j.parameter.server.model;
    exports org.nd4j.parameterserver;
    exports org.nd4j.parameterserver.updater;
    exports org.nd4j.parameterserver.updater.storage;
    exports org.nd4j.parameterserver.util;
}
