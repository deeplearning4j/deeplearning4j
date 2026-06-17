open module deeplearning4j.vertx {
    requires io.vertx.core;
    requires io.vertx.web;
    requires deeplearning4j.core;
    requires deeplearning4j.nn;
    requires deeplearning4j.ui.model;
    requires freemarker;
    requires jcommander;
    requires jakarta.xml.bind;
    requires nd4j.api;
    requires nd4j.webjar;
    exports org.deeplearning4j.ui;
    exports org.deeplearning4j.ui.api;
    exports org.deeplearning4j.ui.i18n;
    exports org.deeplearning4j.ui.module;
    exports org.deeplearning4j.ui.module.convolutional;
    exports org.deeplearning4j.ui.module.defaultModule;
    exports org.deeplearning4j.ui.module.remote;
    exports org.deeplearning4j.ui.module.train;
    exports org.deeplearning4j.ui.module.tsne;
}
