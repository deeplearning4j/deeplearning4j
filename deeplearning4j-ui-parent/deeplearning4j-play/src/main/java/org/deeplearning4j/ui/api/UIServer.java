package org.deeplearning4j.ui.api;

import org.deeplearning4j.ui.play.PlayUIServer;
import org.deeplearning4j.ui.storage.StatsStorage;

import java.util.List;

/**
 * Created by Alex on 08/10/2016.
 */
public abstract class UIServer {

    private static UIServer uiServer;

    public static synchronized UIServer getInstance(){
        if(uiServer == null) uiServer = new PlayUIServer();
        return uiServer;
    }


    public abstract void attach(StatsStorage statsStorage);

    public abstract void detach(StatsStorage statsStorage);

    public abstract boolean isAttached(StatsStorage statsStorage);

    public abstract List<StatsStorage> getStatsStorageInstances();

}
