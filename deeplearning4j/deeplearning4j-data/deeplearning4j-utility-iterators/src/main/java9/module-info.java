module deeplearning4j.utility.iterators {
    requires commons.io;
    requires guava;

    requires transitive nd4j.api;
    requires transitive nd4j.common;
    requires transitive slf4j.api;

    exports org.deeplearning4j.datasets.iterator;
    exports org.deeplearning4j.datasets.iterator.callbacks;
    exports org.deeplearning4j.datasets.iterator.file;
    exports org.deeplearning4j.datasets.iterator.loader;
    exports org.deeplearning4j.datasets.iterator.parallel;
    exports org.deeplearning4j.datasets.iterator.utilty;

}
