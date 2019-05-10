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

package org.deeplearning4j.arbiter.server;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.arbiter.evaluator.multilayer.ClassificationEvaluator;
import org.deeplearning4j.arbiter.evaluator.multilayer.RegressionDataEvaluator;
import org.deeplearning4j.arbiter.optimize.api.data.DataSetIteratorFactoryProvider;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.LocalOptimizationRunner;
import org.deeplearning4j.arbiter.scoring.RegressionValue;
import org.deeplearning4j.arbiter.server.cli.NeuralNetTypeValidator;
import org.deeplearning4j.arbiter.server.cli.ProblemTypeValidator;
import org.deeplearning4j.arbiter.task.ComputationGraphTaskCreator;
import org.deeplearning4j.arbiter.task.MultiLayerNetworkTaskCreator;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

/**
 *   Options:
 *     --dataSetIteratorClass
       --modelSavePath
 Default: /tmp
 *     --neuralNetType
       --optimizationConfigPath
       --problemType
           Default: classification
 --regressionType



 @author Adam Gibson
 */
public class ArbiterCliRunner {
    @Parameter(names = {"--modelSavePath"})
    private String modelSavePath = System.getProperty("java.io.tmpdir");
    @Parameter(names = {"--optimizationConfigPath"})
    private String optimizationConfigPath = null;
    @Parameter(names = {"--problemType"},validateWith = ProblemTypeValidator.class)
    private String problemType = CLASSIFICATION;
    @Parameter(names = {"--regressionType"})
    private String regressionType = null;
    @Parameter(names = {"--dataSetIteratorClass"},required = true)
    private String dataSetIteratorClass = null;
    @Parameter(names = {"--neuralNetType"},required = true,validateWith = NeuralNetTypeValidator.class)
    private String neuralNetType = null;

    public final static String CLASSIFICATION = "classification";
    public final static String REGRESSION = "regression";


    public final static String COMP_GRAPH = "compgraph";
    public final static String MULTI_LAYER_NETWORK = "multilayernetwork";

    public void runMain(String...args) throws Exception {
        JCommander jcmdr = new JCommander(this);

        try {
            jcmdr.parse(args);
        } catch(ParameterException e) {
            System.err.println(e.getMessage());
            //User provides invalid input -> print the usage info
            jcmdr.usage();
            try{ Thread.sleep(500); } catch(Exception e2){ }
            System.exit(1);
        }

        Map<String,Object> commands = new HashMap<>();
        commands.put(DataSetIteratorFactoryProvider.FACTORY_KEY,dataSetIteratorClass);

        File f = new File(modelSavePath);

        if(f.exists()) f.delete();
        f.mkdir();
        f.deleteOnExit();

        if(problemType.equals(REGRESSION)) {
            if(neuralNetType.equals(COMP_GRAPH)) {
                OptimizationConfiguration configuration
                        = OptimizationConfiguration.fromJson(
                        FileUtils.readFileToString(new File(optimizationConfigPath)));

                IOptimizationRunner runner
                        = new LocalOptimizationRunner(configuration, new ComputationGraphTaskCreator(
                        new RegressionDataEvaluator(RegressionValue.valueOf(regressionType),commands)));
                runner.execute();
            }
            else if(neuralNetType.equals(MULTI_LAYER_NETWORK)) {
                OptimizationConfiguration configuration = OptimizationConfiguration.
                        fromJson(FileUtils.readFileToString(new File(optimizationConfigPath)));

                IOptimizationRunner runner
                        = new LocalOptimizationRunner(
                        configuration,
                        new MultiLayerNetworkTaskCreator(
                                new RegressionDataEvaluator(
                                        RegressionValue.valueOf(regressionType),
                                        commands)));
                runner.execute();
            }
        }

        else if(problemType.equals(CLASSIFICATION)) {
            if(neuralNetType.equals(COMP_GRAPH)) {
                OptimizationConfiguration configuration
                        = OptimizationConfiguration.fromJson(FileUtils.readFileToString(new File(optimizationConfigPath)));

                IOptimizationRunner runner
                        = new LocalOptimizationRunner(
                        configuration,new ComputationGraphTaskCreator(new ClassificationEvaluator()));

                runner.execute();
            }
            else if(neuralNetType.equals(MULTI_LAYER_NETWORK)) {
                OptimizationConfiguration configuration = OptimizationConfiguration
                        .fromJson(FileUtils.readFileToString(new File(optimizationConfigPath)));

                IOptimizationRunner runner
                        = new LocalOptimizationRunner(configuration,
                        new MultiLayerNetworkTaskCreator(
                                new ClassificationEvaluator())
                );

                runner.execute();
            }
        }
    }
    public static void main(String...args) throws Exception {
        new ArbiterCliRunner().runMain(args);
    }

}
