open module deeplearning4j.ui.model {
    requires nd4j.api;
    requires nd4j.cpu.api;
    requires deeplearning4j.core;
    requires deeplearning4j.nn;
    requires mapdb;
    requires org.xerial.sqlitejdbc;
    requires lz4.pure.java;
    requires java.sql;
    requires io.aeron.all;
    exports org.deeplearning4j.ui.model;
    exports org.deeplearning4j.ui.model.activation;
    exports org.deeplearning4j.ui.model.nearestneighbors;
    exports org.deeplearning4j.ui.model.nearestneighbors.word2vec;
    exports org.deeplearning4j.ui.model.renders;
    exports org.deeplearning4j.ui.model.stats;
    exports org.deeplearning4j.ui.model.stats.api;
    exports org.deeplearning4j.ui.model.stats.impl;
    exports org.deeplearning4j.ui.model.stats.impl.java;
    exports org.deeplearning4j.ui.model.stats.sbe;
    exports org.deeplearning4j.ui.model.storage;
    exports org.deeplearning4j.ui.model.storage.impl;
    exports org.deeplearning4j.ui.model.storage.mapdb;
    exports org.deeplearning4j.ui.model.storage.sqlite;
    exports org.deeplearning4j.ui.model.weights;
    exports org.deeplearning4j.ui.model.weights.beans;
}
