open module deeplearning4j.parallel.wrapper {
    requires deeplearning4j.utility.iterators;
    requires guava;
    requires jcommander;
    requires resources;
    requires slf4j.api;
    requires deeplearning4j.core;
    requires deeplearning4j.nn;
    requires nd4j.api;
    requires nd4j.common;
    exports org.deeplearning4j.parallelism;
    exports org.deeplearning4j.parallelism.factory;
    exports org.deeplearning4j.parallelism.inference;
    exports org.deeplearning4j.parallelism.inference.observers;
    exports org.deeplearning4j.parallelism.main;
    exports org.deeplearning4j.parallelism.trainer;
}
