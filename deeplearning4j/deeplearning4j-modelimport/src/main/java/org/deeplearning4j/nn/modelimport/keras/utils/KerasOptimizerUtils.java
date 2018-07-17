/*
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
package org.deeplearning4j.nn.modelimport.keras.utils;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.nd4j.linalg.learning.config.*;
import org.nd4j.linalg.schedule.InverseSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

import java.util.Map;

/**
 * Utility functionality for keras optimizers
 *
 * @author Max Pumperla
 */
@Slf4j
public class KerasOptimizerUtils {
    /**
     * Map Keras optimizer to DL4J IUpdater.
     *
     * @param optimizerConfig Optimizer configuration map
     * @return DL4J IUpdater instance
     */
    public static IUpdater mapOptimizer(Map<String, Object> optimizerConfig)
            throws UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {

        System.out.println(optimizerConfig);

        if (!optimizerConfig.containsKey("class_name")) {
            throw new InvalidKerasConfigurationException("Optimizer config does not contain a name field.");
        }
        String optimizerName = (String) optimizerConfig.get("class_name");

        if (!optimizerConfig.containsKey("config"))
            throw new InvalidKerasConfigurationException("Field config missing from layer config");
        Map<String, Object> optimizerParameters = (Map<String, Object>) optimizerConfig.get("config");

        IUpdater dl4jOptimizer;


        switch (optimizerName) {
            case "Adam": {
                double lr = (double) optimizerParameters.get("lr");
                double beta1 = (double) optimizerParameters.get("beta_1");
                double beta2 = (double) optimizerParameters.get("beta_2");
                double epsilon = (double) optimizerParameters.get("epsilon");
                double decay = (double) optimizerParameters.get("decay");

                dl4jOptimizer = new Adam.Builder()
                        .beta1(beta1).beta2(beta2)
                        .epsilon(epsilon).learningRate(lr)
                        .learningRateSchedule(new InverseSchedule(ScheduleType.ITERATION, 1, decay, 1))
                        .build();
                break;
            }
            case "Adadelta": {
                double rho = (double) optimizerParameters.get("rho");
                double epsilon = (double) optimizerParameters.get("epsilon");
                // double decay = (double) optimizerParameters.get("decay"); No decay in DL4J Adadelta

                dl4jOptimizer = new AdaDelta.Builder()
                        .epsilon(epsilon).rho(rho)
                        .build();
                break;
            }
            case "Adgrad": {
                double lr = (double) optimizerParameters.get("lr");
                double epsilon = (double) optimizerParameters.get("epsilon");
                double decay = (double) optimizerParameters.get("decay");

                dl4jOptimizer = new AdaGrad.Builder()
                        .epsilon(epsilon).learningRate(lr)
                        .learningRateSchedule(new InverseSchedule(ScheduleType.ITERATION, 1, decay, 1))
                        .build();
                break;
            }
            case "Adamax": {
                double lr = (double) optimizerParameters.get("lr");
                double beta1 = (double) optimizerParameters.get("beta_1");
                double beta2 = (double) optimizerParameters.get("beta_2");
                double epsilon = (double) optimizerParameters.get("epsilon");

                dl4jOptimizer = new AdaMax(lr, beta1, beta2, epsilon);
                break;
            }
            case "Nadam": {
                double lr = (double) optimizerParameters.get("lr");
                double beta1 = (double) optimizerParameters.get("beta_1");
                double beta2 = (double) optimizerParameters.get("beta_2");
                double epsilon = (double) optimizerParameters.get("epsilon");
                double decay = (double) optimizerParameters.get("decay");

                dl4jOptimizer = new Nadam.Builder()
                        .beta1(beta1).beta2(beta2)
                        .epsilon(epsilon).learningRate(lr)
                        .learningRateSchedule(new InverseSchedule(ScheduleType.ITERATION, 1, decay, 1))
                        .build();
                break;
            }
            case "SGD": {
                double lr = (double) optimizerParameters.get("lr");
                double momentum = (double) optimizerParameters.get("epsilon");
                double decay = (double) optimizerParameters.get("decay");

                dl4jOptimizer = new Nesterovs.Builder()
                        .momentum(momentum).learningRate(lr)
                        .learningRateSchedule(new InverseSchedule(ScheduleType.ITERATION, 1, decay, 1))
                        .build();
                break;
            }
            case "RMSprop": {
                double lr = (double) optimizerParameters.get("lr");
                double rho = (double) optimizerParameters.get("rho");
                double epsilon = (double) optimizerParameters.get("epsilon");
                double decay = (double) optimizerParameters.get("decay");

                dl4jOptimizer = new RmsProp.Builder()
                        .epsilon(epsilon).rmsDecay(rho).learningRate(lr)
                        .learningRateSchedule(new InverseSchedule(ScheduleType.ITERATION, 1, decay, 1))
                        .build();
                break;
            }
            default:
                throw new UnsupportedKerasConfigurationException("Optimizer with name " + optimizerName + "can not be" +
                        "matched to a DL4J optimizer. Note that custom TFOptimizers are not supported by model import");
        }

        return dl4jOptimizer;

    }
}
