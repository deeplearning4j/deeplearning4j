package org.deeplearning4j.ui.stats.impl;

import lombok.AllArgsConstructor;
import lombok.Builder;
import org.deeplearning4j.ui.stats.api.StatsType;
import org.deeplearning4j.ui.stats.api.StatsUpdateConfiguration;

/**
 * Created by Alex on 07/10/2016.
 */
@Builder @AllArgsConstructor
public class DefaultStatsUpdateConfiguration implements StatsUpdateConfiguration {

    private int reportingFrequency = 1;
    private boolean collectPerformanceStats = true;
    private boolean collectMemoryStats = true;
    private boolean collectGarbageCollectionStats = true;
    private boolean collectLearningRates = true;
    private boolean collectHistogramsParameters = true;
    private boolean collectHistogramsUpdates = true;
    private boolean isCollectHistogramsActivations = true;
    private int numHistogramBins = 20;
    private boolean collectMeanParameters = true;
    private boolean collectMeanUpdates = true;
    private boolean collectMeanActivations = true;
    private boolean collectStdevParameters = true;
    private boolean collectStdevUpdates = true;
    private boolean collectStdevActivations = true;
    private boolean collectMeanMagnitudesParameters = true;
    private boolean collectMeanMagnitudesUpdates = true;
    private boolean collectMeanMagnitudesActivations = true;


    @Override
    public int reportingFrequency() {
        return reportingFrequency;
    }

    @Override
    public boolean collectPerformanceStats() {
        return collectPerformanceStats;
    }

    @Override
    public boolean collectMemoryStats() {
        return collectMemoryStats;
    }

    @Override
    public boolean collectGarbageCollectionStats() {
        return collectGarbageCollectionStats;
    }

    @Override
    public boolean collectLearningRates() {
        return collectLearningRates;
    }

    @Override
    public boolean collectHistograms(StatsType type) {
        switch (type){
            case Parameters:
                return collectHistogramsParameters;
            case Updates:
                return collectHistogramsUpdates;
            case Activations:
                return isCollectHistogramsActivations;
        }
        return false;
    }

    @Override
    public int numHistogramBins(StatsType type) {
        return numHistogramBins;
    }

    @Override
    public boolean collectMean(StatsType type) {
        switch(type){
            case Parameters:
                return collectMeanParameters;
            case Updates:
                return collectMeanUpdates;
            case Activations:
                return collectMeanActivations;
        }
        return false;
    }

    @Override
    public boolean collectStdev(StatsType type) {
        switch(type){
            case Parameters:
                return collectStdevParameters;
            case Updates:
                return collectStdevUpdates;
            case Activations:
                return collectStdevActivations;
        }
        return false;
    }

    @Override
    public boolean collectMeanMagnitudes(StatsType type) {
        switch(type){
            case Parameters:
                return collectMeanMagnitudesParameters;
            case Updates:
                return collectMeanMagnitudesUpdates;
            case Activations:
                return collectMeanMagnitudesActivations;
        }
        return false;
    }
}
