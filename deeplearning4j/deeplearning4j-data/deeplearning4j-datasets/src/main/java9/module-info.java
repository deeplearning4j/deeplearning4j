open module deeplearning4j.datasets {
    requires commons.io;
    requires lombok;
    requires nd4j.common;
    requires slf4j.api;
    requires datavec.api;
    requires datavec.data.image;
    requires deeplearning4j.datavec.iterators;
    requires nd4j.api;
    requires resources;
    exports org.deeplearning4j.datasets.base;
    exports org.deeplearning4j.datasets.fetchers;
    exports org.deeplearning4j.datasets.iterator.impl;
    exports org.deeplearning4j.datasets.mnist;
}
