open module nd4j.parameter.server.client {
    requires nd4j.parameter.server.model;
    requires slf4j.api;
    requires unirest.java;
    requires io.aeron.all;
    requires jackson;
    requires nd4j.aeron;
    requires nd4j.api;
    exports org.nd4j.parameterserver.client;
}
