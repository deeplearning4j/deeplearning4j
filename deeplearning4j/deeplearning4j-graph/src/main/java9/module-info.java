open module deeplearning4j.graph {
    requires commons.io;
    requires commons.math3;
    requires nd4j.common;
    requires slf4j.api;
    requires threadly;
    requires nd4j.api;
    exports org.deeplearning4j.graph.api;
    exports org.deeplearning4j.graph.data;
    exports org.deeplearning4j.graph.data.impl;
    exports org.deeplearning4j.graph.exception;
    exports org.deeplearning4j.graph.graph;
    exports org.deeplearning4j.graph.iterator;
    exports org.deeplearning4j.graph.iterator.parallel;
    exports org.deeplearning4j.graph.models;
    exports org.deeplearning4j.graph.models.deepwalk;
    exports org.deeplearning4j.graph.models.embeddings;
    exports org.deeplearning4j.graph.models.loader;
    exports org.deeplearning4j.graph.vertexfactory;
}
