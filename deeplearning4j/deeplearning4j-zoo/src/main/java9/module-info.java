open module deeplearning4j.zoo {
    requires commons.io;
    requires jackson;
    requires resources;
    requires slf4j.api;
    requires deeplearning4j.nn;
    requires nd4j.api;
    requires nd4j.common;
    exports org.deeplearning4j.zoo;
    exports org.deeplearning4j.zoo.model;
    exports org.deeplearning4j.zoo.model.helper;
    exports org.deeplearning4j.zoo.util;
    exports org.deeplearning4j.zoo.util.darknet;
    exports org.deeplearning4j.zoo.util.imagenet;
}
