open module deeplearning4j.scaleout.parallelwrapper.parameter.server {
    requires io.aeron.all;
    requires nd4j.aeron;
    requires nd4j.parameter.server;
    requires nd4j.parameter.server.node;
    requires slf4j.api;
    requires deeplearning4j.nn;
    requires deeplearning4j.parallel.wrapper;
    requires nd4j.api;
    requires nd4j.parameter.server.client;
    exports org.deeplearning4j.parallelism.parameterserver;
}
