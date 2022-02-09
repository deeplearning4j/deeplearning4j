open module datavec.local {
    requires arrow.memory.core;
    requires commons.io;
    requires datavec.arrow;
    requires guava;
    requires protonpack;
    requires slf4j.api;
    requires datavec.api;
    requires nd4j.api;
    requires nd4j.common;
    exports org.datavec.local.transforms;
    exports org.datavec.local.transforms.analysis.aggregate;
    exports org.datavec.local.transforms.analysis.histogram;
    exports org.datavec.local.transforms.functions;
    exports org.datavec.local.transforms.functions.data;
    exports org.datavec.local.transforms.join;
    exports org.datavec.local.transforms.misc;
    exports org.datavec.local.transforms.misc.comparator;
    exports org.datavec.local.transforms.rank;
    exports org.datavec.local.transforms.reduce;
    exports org.datavec.local.transforms.sequence;
    exports org.datavec.local.transforms.transform;
    exports org.datavec.local.transforms.transform.filter;
}
