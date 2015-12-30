package org.deeplearning4j.nn.updater.aggregate;

import org.deeplearning4j.nn.api.Updater;

import java.io.Serializable;

public interface UpdaterAggregator extends Serializable {

    public void aggregate(Updater updater);

    public void merge(UpdaterAggregator aggregator);

    public Updater getUpdater();

}
