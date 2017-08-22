package org.deeplearning4j.ui.stats.impl;

import lombok.AllArgsConstructor;
import org.deeplearning4j.ui.stats.api.StatsType;
import org.deeplearning4j.ui.stats.api.StatsUpdateConfiguration;

/**
 * Created by Alex on 07/10/2016.
 */
@AllArgsConstructor
public class DefaultStatsUpdateConfiguration implements StatsUpdateConfiguration {

    public static final int DEFAULT_REPORTING_FREQUENCY = 10;

    private int reportingFrequency = DEFAULT_REPORTING_FREQUENCY;
    private boolean collectPerformanceStats = true;
    private boolean collectMemoryStats = true;
    private boolean collectGarbageCollectionStats = true;
    private boolean collectLearningRates = true;
    private boolean collectHistogramsParameters = true;
    private boolean collectHistogramsGradients = true;
    private boolean collectHistogramsUpdates = true;
    private boolean collectHistogramsActivations = true;
    private int numHistogramBins = 20;
    private boolean collectMeanParameters = true;
    private boolean collectMeanGradients = true;
    private boolean collectMeanUpdates = true;
    private boolean collectMeanActivations = true;
    private boolean collectStdevParameters = true;
    private boolean collectStdevGradients = true;
    private boolean collectStdevUpdates = true;
    private boolean collectStdevActivations = true;
    private boolean collectMeanMagnitudesParameters = true;
    private boolean collectMeanMagnitudesGradients = true;
    private boolean collectMeanMagnitudesUpdates = true;
    private boolean collectMeanMagnitudesActivations = true;

    private DefaultStatsUpdateConfiguration(Builder b) {
        this.reportingFrequency = b.reportingFrequency;
        this.collectPerformanceStats = b.collectPerformanceStats;
        this.collectMemoryStats = b.collectMemoryStats;
        this.collectGarbageCollectionStats = b.collectGarbageCollectionStats;
        this.collectLearningRates = b.collectLearningRates;
        this.collectHistogramsParameters = b.collectHistogramsParameters;
        this.collectHistogramsGradients = b.collectHistogramsGradients;
        this.collectHistogramsUpdates = b.collectHistogramsUpdates;
        this.collectHistogramsActivations = b.collectHistogramsActivations;
        this.numHistogramBins = b.numHistogramBins;
        this.collectMeanParameters = b.collectMeanParameters;
        this.collectMeanGradients = b.collectMeanGradients;
        this.collectMeanUpdates = b.collectMeanUpdates;
        this.collectMeanActivations = b.collectMeanActivations;
        this.collectStdevParameters = b.collectStdevParameters;
        this.collectStdevGradients = b.collectStdevGradients;
        this.collectStdevUpdates = b.collectStdevUpdates;
        this.collectStdevActivations = b.collectStdevActivations;
        this.collectMeanMagnitudesParameters = b.collectMeanMagnitudesParameters;
        this.collectMeanMagnitudesGradients = b.collectMeanMagnitudesGradients;
        this.collectMeanMagnitudesUpdates = b.collectMeanMagnitudesUpdates;
        this.collectMeanMagnitudesActivations = b.collectMeanMagnitudesActivations;
    }

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
        switch (type) {
            case Parameters:
                return collectHistogramsParameters;
            case Gradients:
                return collectStdevGradients;
            case Updates:
                return collectHistogramsUpdates;
            case Activations:
                return collectHistogramsActivations;
        }
        return false;
    }

    @Override
    public int numHistogramBins(StatsType type) {
        return numHistogramBins;
    }

    @Override
    public boolean collectMean(StatsType type) {
        switch (type) {
            case Parameters:
                return collectMeanParameters;
            case Gradients:
                return collectMeanGradients;
            case Updates:
                return collectMeanUpdates;
            case Activations:
                return collectMeanActivations;
        }
        return false;
    }

    @Override
    public boolean collectStdev(StatsType type) {
        switch (type) {
            case Parameters:
                return collectStdevParameters;
            case Gradients:
                return collectStdevGradients;
            case Updates:
                return collectStdevUpdates;
            case Activations:
                return collectStdevActivations;
        }
        return false;
    }

    @Override
    public boolean collectMeanMagnitudes(StatsType type) {
        switch (type) {
            case Parameters:
                return collectMeanMagnitudesParameters;
            case Gradients:
                return collectMeanMagnitudesGradients;
            case Updates:
                return collectMeanMagnitudesUpdates;
            case Activations:
                return collectMeanMagnitudesActivations;
        }
        return false;
    }

    public static class Builder {
        private int reportingFrequency = DEFAULT_REPORTING_FREQUENCY;
        private boolean collectPerformanceStats = true;
        private boolean collectMemoryStats = true;
        private boolean collectGarbageCollectionStats = true;
        private boolean collectLearningRates = true;
        private boolean collectHistogramsParameters = true;
        private boolean collectHistogramsGradients = true;
        private boolean collectHistogramsUpdates = true;
        private boolean collectHistogramsActivations = true;
        private int numHistogramBins = 20;
        private boolean collectMeanParameters = true;
        private boolean collectMeanGradients = true;
        private boolean collectMeanUpdates = true;
        private boolean collectMeanActivations = true;
        private boolean collectStdevParameters = true;
        private boolean collectStdevGradients = true;
        private boolean collectStdevUpdates = true;
        private boolean collectStdevActivations = true;
        private boolean collectMeanMagnitudesParameters = true;
        private boolean collectMeanMagnitudesGradients = true;
        private boolean collectMeanMagnitudesUpdates = true;
        private boolean collectMeanMagnitudesActivations = true;

        public Builder reportingFrequency(int reportingFrequency) {
            this.reportingFrequency = reportingFrequency;
            return this;
        }

        public Builder collectPerformanceStats(boolean collectPerformanceStats) {
            this.collectPerformanceStats = collectPerformanceStats;
            return this;
        }

        public Builder collectMemoryStats(boolean collectMemoryStats) {
            this.collectMemoryStats = collectMemoryStats;
            return this;
        }

        public Builder collectGarbageCollectionStats(boolean collectGarbageCollectionStats) {
            this.collectGarbageCollectionStats = collectGarbageCollectionStats;
            return this;
        }

        public Builder collectLearningRates(boolean collectLearningRates) {
            this.collectLearningRates = collectLearningRates;
            return this;
        }

        public Builder collectHistogramsParameters(boolean collectHistogramsParameters) {
            this.collectHistogramsParameters = collectHistogramsParameters;
            return this;
        }

        public Builder collectHistogramsGradients(boolean collectHistogramsGradients) {
            this.collectHistogramsGradients = collectHistogramsGradients;
            return this;
        }

        public Builder collectHistogramsUpdates(boolean collectHistogramsUpdates) {
            this.collectHistogramsUpdates = collectHistogramsUpdates;
            return this;
        }

        public Builder collectHistogramsActivations(boolean isCollectHistogramsActivations) {
            this.collectHistogramsActivations = isCollectHistogramsActivations;
            return this;
        }

        public Builder numHistogramBins(int numHistogramBins) {
            this.numHistogramBins = numHistogramBins;
            return this;
        }

        public Builder collectMeanParameters(boolean collectMeanParameters) {
            this.collectMeanParameters = collectMeanParameters;
            return this;
        }

        public Builder collectMeanGradients(boolean collectMeanGradients) {
            this.collectMeanGradients = collectMeanGradients;
            return this;
        }

        public Builder collectMeanUpdates(boolean collectMeanUpdates) {
            this.collectMeanUpdates = collectMeanUpdates;
            return this;
        }

        public Builder collectMeanActivations(boolean collectMeanActivations) {
            this.collectMeanActivations = collectMeanActivations;
            return this;
        }

        public Builder collectStdevParameters(boolean collectStdevParameters) {
            this.collectStdevParameters = collectStdevParameters;
            return this;
        }

        public Builder collectStdevGradients(boolean collectStdevGradients) {
            this.collectStdevGradients = collectStdevGradients;
            return this;
        }

        public Builder collectStdevUpdates(boolean collectStdevUpdates) {
            this.collectStdevUpdates = collectStdevUpdates;
            return this;
        }

        public Builder collectStdevActivations(boolean collectStdevActivations) {
            this.collectStdevActivations = collectStdevActivations;
            return this;
        }

        public Builder collectMeanMagnitudesParameters(boolean collectMeanMagnitudesParameters) {
            this.collectMeanMagnitudesParameters = collectMeanMagnitudesParameters;
            return this;
        }

        public Builder collectMeanMagnitudesGradients(boolean collectMeanMagnitudesGradients) {
            this.collectMeanMagnitudesGradients = collectMeanMagnitudesGradients;
            return this;
        }

        public Builder collectMeanMagnitudesUpdates(boolean collectMeanMagnitudesUpdates) {
            this.collectMeanMagnitudesUpdates = collectMeanMagnitudesUpdates;
            return this;
        }

        public Builder collectMeanMagnitudesActivations(boolean collectMeanMagnitudesActivations) {
            this.collectMeanMagnitudesActivations = collectMeanMagnitudesActivations;
            return this;
        }

        public DefaultStatsUpdateConfiguration build() {
            return new DefaultStatsUpdateConfiguration(this);
        }
    }
}
