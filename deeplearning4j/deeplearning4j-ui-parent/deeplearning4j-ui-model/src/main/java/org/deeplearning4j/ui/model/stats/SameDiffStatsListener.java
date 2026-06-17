/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.ui.model.stats;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.core.storage.StatsStorageRouter;
import org.deeplearning4j.ui.model.stats.api.StatsReport;
import org.deeplearning4j.ui.model.stats.api.StatsType;
import org.deeplearning4j.ui.model.stats.impl.SbeStatsInitializationReport;
import org.deeplearning4j.ui.model.stats.impl.SbeStatsReport;
import org.deeplearning4j.ui.model.storage.impl.SbeStorageMetaData;
import org.nd4j.autodiff.listeners.BaseListener;
import org.nd4j.autodiff.listeners.Loss;
import org.nd4j.autodiff.listeners.Operation;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.lang.management.ManagementFactory;
import java.lang.management.OperatingSystemMXBean;
import java.lang.management.RuntimeMXBean;
import java.util.*;

/**
 * SameDiff training listener that bridges to the DL4J UI training dashboard.
 * Collects training scores, gradient statistics (mean, stdev, mean magnitude),
 * and parameter statistics, writing them as {@link SbeStatsReport} updates
 * to a {@link StatsStorageRouter} so the existing TrainModule UI can display them.
 *
 * Usage:
 * <pre>
 *     InMemoryStatsStorage storage = new InMemoryStatsStorage();
 *     UIServer uiServer = UIServer.getInstance();
 *     uiServer.attach(storage);
 *
 *     SameDiffStatsListener listener = new SameDiffStatsListener(storage);
 *     sd.setListeners(listener);
 *     sd.fit(dataSet, numEpochs);
 * </pre>
 */
@Slf4j
public class SameDiffStatsListener extends BaseListener {

    private static final String TYPE_ID = BaseStatsListener.TYPE_ID;

    private final StatsStorageRouter router;
    private final int reportFrequency;
    private final String sessionID;
    private final String workerID;

    private boolean initialized = false;
    private final Map<String, Double> gradientMean = new LinkedHashMap<>();
    private final Map<String, Double> gradientStdev = new LinkedHashMap<>();
    private final Map<String, Double> gradientMeanMag = new LinkedHashMap<>();

    /**
     * Create a SameDiffStatsListener that reports every iteration.
     *
     * @param router the stats storage router to write updates to
     */
    public SameDiffStatsListener(StatsStorageRouter router) {
        this(router, 1);
    }

    /**
     * Create a SameDiffStatsListener that reports every N iterations.
     *
     * @param router             the stats storage router to write updates to
     * @param reportFrequency    how often (in iterations) to report stats
     */
    public SameDiffStatsListener(StatsStorageRouter router, int reportFrequency) {
        this.router = router;
        this.reportFrequency = Math.max(1, reportFrequency);
        this.sessionID = UUID.randomUUID().toString();
        this.workerID = getJvmUid() + "_" + Thread.currentThread().getId();
    }

    @Override
    public boolean isActive(Operation operation) {
        return operation == Operation.TRAINING;
    }

    @Override
    public void epochStart(SameDiff sd, At at) {
        if (!initialized) {
            doInit(sd);
        }
    }

    @Override
    public void preUpdate(SameDiff sd, At at, Variable v, INDArray update) {
        if (update == null) return;
        String name = v.getName();
        gradientMean.put(name, update.meanNumber().doubleValue());
        gradientStdev.put(name, update.stdNumber().doubleValue());
        double meanMag = update.norm1Number().doubleValue() / Math.max(update.length(), 1);
        gradientMeanMag.put(name, meanMag);
    }

    @Override
    public void iterationDone(SameDiff sd, At at, org.nd4j.linalg.dataset.api.MultiDataSet dataSet, Loss loss) {
        if (!initialized) {
            doInit(sd);
        }

        int iteration = at.iteration();
        if (iteration % reportFrequency != 0) {
            return;
        }

        SbeStatsReport report = new SbeStatsReport();
        report.reportIDs(sessionID, TYPE_ID, workerID, System.currentTimeMillis());
        report.reportIterationCount(iteration);

        // Score
        if (loss != null) {
            double totalLoss = 0;
            for (double l : loss.getLosses()) {
                totalLoss += l;
            }
            report.reportScore(totalLoss);
        }

        // Gradient stats
        if (!gradientMean.isEmpty()) {
            report.reportMean(StatsType.Gradients, new LinkedHashMap<>(gradientMean));
            report.reportStdev(StatsType.Gradients, new LinkedHashMap<>(gradientStdev));
            report.reportMeanMagnitudes(StatsType.Gradients, new LinkedHashMap<>(gradientMeanMag));
            gradientMean.clear();
            gradientStdev.clear();
            gradientMeanMag.clear();
        }

        // Parameter stats
        Map<String, Double> paramMean = new LinkedHashMap<>();
        Map<String, Double> paramStdev = new LinkedHashMap<>();
        Map<String, Double> paramMeanMag = new LinkedHashMap<>();
        for (SDVariable var : sd.variables()) {
            if (var.getVariableType() == VariableType.VARIABLE) {
                try {
                    INDArray arr = var.getArr();
                    if (arr != null) {
                        paramMean.put(var.name(), arr.meanNumber().doubleValue());
                        paramStdev.put(var.name(), arr.stdNumber().doubleValue());
                        paramMeanMag.put(var.name(), arr.norm1Number().doubleValue() / Math.max(arr.length(), 1));
                    }
                } catch (Exception e) {
                    // Variable array may not be available yet
                }
            }
        }
        if (!paramMean.isEmpty()) {
            report.reportMean(StatsType.Parameters, paramMean);
            report.reportStdev(StatsType.Parameters, paramStdev);
            report.reportMeanMagnitudes(StatsType.Parameters, paramMeanMag);
        }

        router.putUpdate(report);
    }

    private void doInit(SameDiff sd) {
        if (initialized) return;
        initialized = true;

        long initTime = System.currentTimeMillis();

        // Collect param names and count
        List<String> paramNames = new ArrayList<>();
        long totalParams = 0;
        for (SDVariable var : sd.variables()) {
            if (var.getVariableType() == VariableType.VARIABLE) {
                paramNames.add(var.name());
                try {
                    long[] shape = var.getShape();
                    if (shape != null) {
                        long count = 1;
                        for (long s : shape) count *= s;
                        totalParams += count;
                    }
                } catch (Exception e) {
                    // Shape may not be known yet
                }
            }
        }

        SbeStatsInitializationReport initReport = new SbeStatsInitializationReport();
        initReport.reportIDs(sessionID, TYPE_ID, workerID, initTime);

        // Model info
        initReport.reportModelInfo(
                "SameDiff",
                "", // no single JSON config for SameDiff
                paramNames.toArray(new String[0]),
                sd.getOps().size(),
                totalParams
        );

        // Software info
        RuntimeMXBean rt = ManagementFactory.getRuntimeMXBean();
        OperatingSystemMXBean os = ManagementFactory.getOperatingSystemMXBean();
        String backend = "";
        try {
            backend = org.nd4j.linalg.factory.Nd4j.getBackend().getClass().getSimpleName();
        } catch (Exception e) {
            // ignore
        }
        initReport.reportSoftwareInfo(
                os.getArch(),
                os.getName(),
                rt.getVmName(),
                rt.getVmVersion(),
                rt.getSpecVersion(),
                backend,
                org.nd4j.linalg.factory.Nd4j.dataType().name(),
                getHostname(),
                getJvmUid(),
                Collections.emptyMap()
        );

        // Hardware info
        int nProc = Runtime.getRuntime().availableProcessors();
        long maxMem = Runtime.getRuntime().maxMemory();
        initReport.reportHardwareInfo(
                nProc,
                0, // numDevices - would need backend-specific query
                maxMem,
                org.nd4j.linalg.factory.Nd4j.getMemoryManager().getCurrentWorkspace() != null ? 0 : 0,
                new long[0],
                new String[0],
                getJvmUid()
        );

        // Storage metadata
        SbeStorageMetaData meta = new SbeStorageMetaData(
                initTime, sessionID, TYPE_ID, workerID,
                SbeStatsInitializationReport.class,
                SbeStatsReport.class
        );

        router.putStorageMetaData(meta);
        router.putStaticInfo(initReport);
    }

    private static String getJvmUid() {
        String name = ManagementFactory.getRuntimeMXBean().getName();
        int idx = name.indexOf('@');
        return idx > 0 ? name.substring(0, idx) : name;
    }

    private static String getHostname() {
        try {
            return java.net.InetAddress.getLocalHost().getHostName();
        } catch (Exception e) {
            return "unknown";
        }
    }
}
