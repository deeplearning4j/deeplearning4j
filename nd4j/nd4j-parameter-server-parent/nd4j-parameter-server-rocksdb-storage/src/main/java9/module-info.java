open module nd4j.parameter.server.rocksdb.storage {
    requires io.aeron.all;
    requires rocksdbjni;
    requires nd4j.aeron;
    exports org.nd4j.parameterserver.updater.storage;
}
