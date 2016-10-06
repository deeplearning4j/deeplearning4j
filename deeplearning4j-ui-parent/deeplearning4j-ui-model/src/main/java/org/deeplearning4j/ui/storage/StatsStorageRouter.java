package org.deeplearning4j.ui.storage;

/**
 * Created by Alex on 07/10/2016.
 */
public interface StatsStorageRouter {
    

    void putMetaData(String sessionID, String typeID, String workerID, Class<?> staticInfoClass, Class<?> updateClass );

    void putStaticInfo(Persistable persistable);    //TODO error handling

    void putUpdate(Persistable persistable);        //TODO error handling


    //TODO async methods

}
