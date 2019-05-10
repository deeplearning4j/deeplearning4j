/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.perf.listener;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import oshi.json.SystemInfo;

import java.util.List;
import java.util.Map;


/**
 * Using {@link SystemInfo} - it logs a json representation
 * of system info using slf4j.
 *
 * @author Adam Gibson
 */

@Slf4j
@Builder
public class SystemInfoPrintListener implements TrainingListener {
    private boolean printOnEpochStart;
    private boolean printOnEpochEnd;
    private boolean printOnForwardPass;
    private boolean printOnBackwardPass;
    private boolean printOnGradientCalculation;


    @Override
    public void iterationDone(Model model, int iteration, int epoch) {

    }

    @Override
    public void onEpochStart(Model model) {
       if(!printOnEpochStart)
           return;

        SystemInfo systemInfo = new SystemInfo();
        log.info("System info on epoch begin: ");
        log.info(systemInfo.toPrettyJSON());
    }

    @Override
    public void onEpochEnd(Model model) {
        if(!printOnEpochEnd)
            return;

        SystemInfo systemInfo = new SystemInfo();
        log.info("System info on epoch end: ");
        log.info(systemInfo.toPrettyJSON());
    }

    @Override
    public void onForwardPass(Model model, List<INDArray> activations) {
        if(!printOnBackwardPass)
            return;

        SystemInfo systemInfo = new SystemInfo();
        log.info("System info on epoch end: ");
        log.info(systemInfo.toPrettyJSON());
    }

    @Override
    public void onForwardPass(Model model, Map<String, INDArray> activations) {
        if(!printOnForwardPass)
            return;

        SystemInfo systemInfo = new SystemInfo();
        log.info("System info on epoch end: ");
        log.info(systemInfo.toPrettyJSON());
    }

    @Override
    public void onGradientCalculation(Model model) {
        if(!printOnGradientCalculation)
            return;

        SystemInfo systemInfo = new SystemInfo();
        log.info("System info on epoch end: ");
        log.info(systemInfo.toPrettyJSON());
    }

    @Override
    public void onBackwardPass(Model model) {
        if(!printOnBackwardPass)
            return;
        SystemInfo systemInfo = new SystemInfo();
        log.info("System info on epoch end: ");
        log.info(systemInfo.toPrettyJSON());
    }
}
