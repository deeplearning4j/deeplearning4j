open module nd4j.common {
    requires commons.io;
    requires org.apache.commons.compress;
    requires org.apache.commons.lang3;
    requires commons.math3;
    requires guava;
    requires jackson;
    requires slf4j.api;
    exports org.nd4j.common.base;
    exports org.nd4j.common.collection;
    exports org.nd4j.common.collections;
    exports org.nd4j.common.config;
    exports org.nd4j.common.function;
    exports org.nd4j.common.holder;
    exports org.nd4j.common.io;
    exports org.nd4j.common.loader;
    exports org.nd4j.common.primitives;
    exports org.nd4j.common.primitives.serde;
    exports org.nd4j.common.tools;
    exports org.nd4j.common.util;
    exports org.nd4j.common.validation;
}
