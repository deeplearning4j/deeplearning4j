package org.deeplearning4j.ui.stats.impl;

import lombok.AllArgsConstructor;
import org.deeplearning4j.ui.stats.api.StatsType;
import org.deeplearning4j.ui.stats.api.StatsUpdateConfiguration;

/**
 * Created by Alex on 07/10/2016.
 */
@AllArgsConstructor
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

    public static DefaultStatsUpdateConfigurationBuilder builder() {
        return new DefaultStatsUpdateConfigurationBuilder();
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

    public static class DefaultStatsUpdateConfigurationBuilder {
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

        DefaultStatsUpdateConfigurationBuilder() {
        }

        public DefaultStatsUpdateConfiguration.DefaultStatsUpdateConfigurationBuilder reportingFrequency(int reportingFrequency) {
            this.reportingFrequency = reportingFrequency;
            return this;
        }

        public DefaultStatsUpdateConfiguration.DefaultStatsUpdateConfigurationBuilder collectPerformanceStats(boolean collectPerformanceStats) {
            this.collectPerformanceStats = collectPerformanceStats;
            return this;
        }

        public DefaultStatsUpdateConfiguration.DefaultStatsUpdateConfigurationBuilder collectMemoryStats(boolean collectMemoryStats) {
            this.collectMemoryStats = collectMemoryStats;
            return this;
        }

        public DefaultStatsUpdateConfiguration.DefaultStatsUpdateConfigurationBuilder collectGarbageCollectionStats(boolean collectGarbageCollectionStats) {
            this.collectGarbageCollectionStats = collectGarbageCollectionStats;
            return this;
        }

        public DefaultStatsUpdateConfiguration.DefaultStatsUpdateConfigurationBuilder collectLearningRates(boolean collectLearningRates) {
            this.collectLearningRates = collectLearningRates;
            return this;
        }

        public DefaultStatsUpdateConfiguration.DefaultStatsUpdateConfigurationBuilder collectHistogramsParameters(boolean collectHistogramsParameters) {
            this.collectHistogramsParameters = collectHistogramsParameters;
            return this;
        }

        public DefaultStatsUpdateConfiguration.DefaultStatsUpdateConfigurationBuilder collectHistogramsUpdates(boolean collectHistogramsUpdates) {
            this.collectHistogramsUpdates = collectHistogramsUpdates;
            return this;
        }

        public DefaultStatsUpdateConfiguration.DefaultStatsUpdateConfigurationBuilder isCollectHistogramsActivations(boolean isCollectHistogramsActivations) {
            this.isCollectHistogramsActivations = isCollectHistogramsActivations;
            return this;
        }

        public DefaultStatsUpdateConfiguration.DefaultStatsUpdateConfigurationBuilder numHistogramBins(int numHistogramBins) {
            this.numHistogramBins = numHistogramBins;
            return this;
        }

        public DefaultStatsUpdateConfiguration.DefaultStatsUpdateConfigurationBuilder collectMeanParameters(boolean collectMeanParameters) {
            this.collectMeanParameters = collectMeanParameters;
            return this;
        }

        public DefaultStatsUpdateConfiguration.DefaultStatsUpdateConfigurationBuilder collectMeanUpdates(boolean collectMeanUpdates) {
            this.collectMeanUpdates = collectMeanUpdates;
            return this;
        }

        public DefaultStatsUpdateConfiguration.DefaultStatsUpdateConfigurationBuilder collectMeanActivations(boolean collectMeanActivations) {
            this.collectMeanActivations = collectMeanActivations;
            return this;
        }

        public DefaultStatsUpdateConfiguration.DefaultStatsUpdateConfigurationBuilder collectStdevParameters(boolean collectStdevParameters) {
            this.collectStdevParameters = collectStdevParameters;
            return this;
        }

        public DefaultStatsUpdateConfiguration.DefaultStatsUpdateConfigurationBuilder collectStdevUpdates(boolean collectStdevUpdates) {
            this.collectStdevUpdates = collectStdevUpdates;
            return this;
        }

        public DefaultStatsUpdateConfiguration.DefaultStatsUpdateConfigurationBuilder collectStdevActivations(boolean collectStdevActivations) {
            this.collectStdevActivations = collectStdevActivations;
            return this;
        }

        public DefaultStatsUpdateConfiguration.DefaultStatsUpdateConfigurationBuilder collectMeanMagnitudesParameters(boolean collectMeanMagnitudesParameters) {
            this.collectMeanMagnitudesParameters = collectMeanMagnitudesParameters;
            return this;
        }

        public DefaultStatsUpdateConfiguration.DefaultStatsUpdateConfigurationBuilder collectMeanMagnitudesUpdates(boolean collectMeanMagnitudesUpdates) {
            this.collectMeanMagnitudesUpdates = collectMeanMagnitudesUpdates;
            return this;
        }

        public DefaultStatsUpdateConfiguration.DefaultStatsUpdateConfigurationBuilder collectMeanMagnitudesActivations(boolean collectMeanMagnitudesActivations) {
            this.collectMeanMagnitudesActivations = collectMeanMagnitudesActivations;
            return this;
        }

        public DefaultStatsUpdateConfiguration build() {
            return new DefaultStatsUpdateConfiguration(reportingFrequency, collectPerformanceStats, collectMemoryStats, collectGarbageCollectionStats, collectLearningRates, collectHistogramsParameters, collectHistogramsUpdates, isCollectHistogramsActivations, numHistogramBins, collectMeanParameters, collectMeanUpdates, collectMeanActivations, collectStdevParameters, collectStdevUpdates, collectStdevActivations, collectMeanMagnitudesParameters, collectMeanMagnitudesUpdates, collectMeanMagnitudesActivations);
        }

        public String toString() {
            return "org.deeplearning4j.ui.stats.impl.DefaultStatsUpdateConfiguration.DefaultStatsUpdateConfigurationBuilder(reportingFrequency=" + this.reportingFrequency + ", collectPerformanceStats=" + this.collectPerformanceStats + ", collectMemoryStats=" + this.collectMemoryStats + ", collectGarbageCollectionStats=" + this.collectGarbageCollectionStats + ", collectLearningRates=" + this.collectLearningRates + ", collectHistogramsParameters=" + this.collectHistogramsParameters + ", collectHistogramsUpdates=" + this.collectHistogramsUpdates + ", isCollectHistogramsActivations=" + this.isCollectHistogramsActivations + ", numHistogramBins=" + this.numHistogramBins + ", collectMeanParameters=" + this.collectMeanParameters + ", collectMeanUpdates=" + this.collectMeanUpdates + ", collectMeanActivations=" + this.collectMeanActivations + ", collectStdevParameters=" + this.collectStdevParameters + ", collectStdevUpdates=" + this.collectStdevUpdates + ", collectStdevActivations=" + this.collectStdevActivations + ", collectMeanMagnitudesParameters=" + this.collectMeanMagnitudesParameters + ", collectMeanMagnitudesUpdates=" + this.collectMeanMagnitudesUpdates + ", collectMeanMagnitudesActivations=" + this.collectMeanMagnitudesActivations + ")";
        }
    }
}
