open module samediff.modelimport.api {
    requires commons.io;
    requires kotlin.logging.jvm;
    requires nd4j.common;
    requires org.apache.commons.lang3;
    requires org.bytedeco.javacpp;
    requires commons.collections4;
    requires guava;
    requires io.github.classgraph;
    requires kotlin.stdlib;
    requires nd4j.api;
    requires protobuf;
    exports org.nd4j.samediff.frameworkimport;
    exports org.nd4j.samediff.frameworkimport.context;
    exports org.nd4j.samediff.frameworkimport.hooks;
    exports org.nd4j.samediff.frameworkimport.hooks.annotations;
    exports org.nd4j.samediff.frameworkimport.ir;
    exports org.nd4j.samediff.frameworkimport.mapper;
    exports org.nd4j.samediff.frameworkimport.opdefs;
    exports org.nd4j.samediff.frameworkimport.optimize;
    exports org.nd4j.samediff.frameworkimport.process;
    exports org.nd4j.samediff.frameworkimport.reflect;
    exports org.nd4j.samediff.frameworkimport.registry;
    exports org.nd4j.samediff.frameworkimport.rule;
    exports org.nd4j.samediff.frameworkimport.rule.attribute;
    exports org.nd4j.samediff.frameworkimport.rule.tensor;
    exports org.nd4j.samediff.frameworkimport.runner;
}
