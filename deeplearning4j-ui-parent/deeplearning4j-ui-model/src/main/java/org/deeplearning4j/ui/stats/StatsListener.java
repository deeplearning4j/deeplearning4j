package org.deeplearning4j.ui.stats;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.IOUtils;
import org.bytedeco.javacpp.Pointer;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.api.storage.StorageMetaData;
import org.deeplearning4j.api.storage.listener.RoutingIterationListener;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.stats.api.*;
import org.deeplearning4j.ui.stats.impl.DefaultStatsInitializationConfiguration;
import org.deeplearning4j.ui.stats.impl.DefaultStatsUpdateConfiguration;
import org.deeplearning4j.ui.stats.impl.SbeStatsInitializationReport;
import org.deeplearning4j.ui.stats.impl.SbeStatsReport;
import org.deeplearning4j.ui.storage.impl.SbeStorageMetaData;
import org.deeplearning4j.util.UIDProvider;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.io.InputStream;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.lang.management.OperatingSystemMXBean;
import java.lang.management.RuntimeMXBean;
import java.lang.reflect.Constructor;
import java.util.*;

/**
 * StatsListener: a general purpose listener for collecting and reporting system and model information.
 * <p>
 * Stats are collected and passed on to a {@link StatsStorageRouter} - for example, for storage and/or displaying in the UI,
 * use {@link org.deeplearning4j.ui.storage.InMemoryStatsStorage} or {@link org.deeplearning4j.ui.storage.FileStatsStorage}.
 *
 * @author Alex Black
 */
@Slf4j
public class StatsListener extends BaseStatsListener {

    /**
     * Create a StatsListener with network information collected at every iteration. Equivalent to {@link #StatsListener(StatsStorageRouter, int)}
     * with {@code listenerFrequency == 1}
     *
     * @param router Where/how to store the calculated stats. For example, {@link org.deeplearning4j.ui.storage.InMemoryStatsStorage} or
     *               {@link org.deeplearning4j.ui.storage.FileStatsStorage}
     */
    public StatsListener(StatsStorageRouter router) {
        this(router, null, null, null, null);
    }

    /**
     * Create a StatsListener with network information collected every n >= 1 time steps
     *
     * @param router            Where/how to store the calculated stats. For example, {@link org.deeplearning4j.ui.storage.InMemoryStatsStorage} or
     *                          {@link org.deeplearning4j.ui.storage.FileStatsStorage}
     * @param listenerFrequency Frequency with which to collect stats information
     */
    public StatsListener(StatsStorageRouter router, int listenerFrequency) {
        this(router, null, new DefaultStatsUpdateConfiguration.Builder().reportingFrequency(listenerFrequency).build(), null, null);
    }

    public StatsListener(StatsStorageRouter router, StatsInitializationConfiguration initConfig, StatsUpdateConfiguration updateConfig,
                             String sessionID, String workerID) {
        super(router, initConfig, updateConfig, sessionID, workerID);
    }

    @Override
    public StatsInitializationReport getNewInitializationReport() {
        return new SbeStatsInitializationReport();
    }

    @Override
    public StatsReport getNewStatsReport() {
        return new SbeStatsReport();
    }

    @Override
    public StorageMetaData getNewStorageMetaData(long initTime, String sessionID, String workerID) {
        return new SbeStorageMetaData(initTime, sessionID, BaseStatsListener.TYPE_ID, workerID, SbeStatsInitializationReport.class,
                SbeStatsReport.class);
    }
}
