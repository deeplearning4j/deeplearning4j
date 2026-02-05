open module nd4j.common.tests {
    requires nd4j.api;
    requires org.junit.jupiter.api;
    requires org.junit.jupiter.engine;
    requires org.junit.jupiter.params;
    requires ch.qos.logback.classic;
    exports org.nd4j.common.tests;
    exports org.nd4j.common.tests.tags;
    exports org.nd4j.linalg;
}
