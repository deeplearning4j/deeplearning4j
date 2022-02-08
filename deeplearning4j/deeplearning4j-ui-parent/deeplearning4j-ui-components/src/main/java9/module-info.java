open module deeplearning4j.ui.components {
    requires commons.io;
    requires freemarker;
    requires jackson;
    requires nd4j.common;
    requires java.desktop;
    exports org.deeplearning4j.ui.api;
    exports org.deeplearning4j.ui.components.chart;
    exports org.deeplearning4j.ui.components.chart.style;
    exports org.deeplearning4j.ui.components.component;
    exports org.deeplearning4j.ui.components.component.style;
    exports org.deeplearning4j.ui.components.decorator;
    exports org.deeplearning4j.ui.components.decorator.style;
    exports org.deeplearning4j.ui.components.table;
    exports org.deeplearning4j.ui.components.table.style;
    exports org.deeplearning4j.ui.components.text;
    exports org.deeplearning4j.ui.components.text.style;
    exports org.deeplearning4j.ui.standalone;
}
