package org.deeplearning4j.api.storage;

import java.io.Serializable;

/**
 * Created by Alex on 11/10/2016.
 */
public interface StorageMetaData extends Persistable {

    long getTimeStamp();

    String getSessionID();

    String getTypeID();

    String getWorkerID();

    String getInitTypeClass();

    String getUpdateTypeClass();

    Serializable getExtraMetaData();

}
