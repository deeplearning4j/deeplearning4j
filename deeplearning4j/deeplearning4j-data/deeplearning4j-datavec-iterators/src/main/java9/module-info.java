open module deeplearning4j.datavec.iterators {
    requires nd4j.common;
    requires org.apache.commons.lang3;
    requires slf4j.api;
    requires datavec.api;
    requires nd4j.api;
    exports org.deeplearning4j.datasets.datavec;
    exports org.deeplearning4j.datasets.datavec.exception;
}
