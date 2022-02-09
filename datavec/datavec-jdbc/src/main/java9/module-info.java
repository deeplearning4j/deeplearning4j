open module datavec.jdbc {
    requires HikariCP.java7;
    requires commons.dbutils;
    requires datavec.api;
    requires java.sql;
    exports org.datavec.jdbc.records.metadata;
    exports org.datavec.jdbc.records.reader.impl.jdbc;
    exports org.datavec.jdbc.util;
}
