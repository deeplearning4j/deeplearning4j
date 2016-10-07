package org.deeplearning4j.ui.storage;

/**
 * StatsStorageRouter is intended to route static info, metadata and updates to a {@link StatsStorage} implementation.
 * For example, a StatsStorageRouter might serialize and send objects over a network
 *
 * @author Alex Black
 */
public interface StatsStorageRouter {


    void putStorageMetaData(StorageMetaData storageMetaData);  //TODO error handling

    void putStaticInfo(Persistable persistable);    //TODO error handling

    void putUpdate(Persistable persistable);        //TODO error handling


    //TODO async methods

}
