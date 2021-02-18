/*
 *  ******************************************************************************
 *  *
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

package org.deeplearning4j.integration;

import lombok.Data;
import org.deeplearning4j.nn.api.Model;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.evaluation.IEvaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.common.primitives.Pair;

import java.io.File;
import java.util.List;
import java.util.Map;

@Data
public abstract class TestCase {

    public enum TestType {
        PRETRAINED, RANDOM_INIT
    }

    //See: readme.md for more details
    protected String testName;                                  //Name of the test, for display purposes
    protected TestType testType;                                //Type of model - from a pretrained model, or a randomly initialized model
    protected boolean testPredictions = true;                   //If true: check the predictions/output. Requires getPredictionsTestData() to be implemented
    protected boolean testGradients = true;                     //If true: check the gradients. Requires getGradientsTestData() to be implemented
    protected boolean testUnsupervisedTraining = false;         //If true: perform unsupervised training. Only applies to layers like autoencoders, VAEs, etc. Requires getUnsupervisedTrainData() to be implemented
    protected boolean testTrainingCurves = true;                //If true: perform training, and compare loss vs. iteration. Requires getTrainingData() method
    protected boolean testParamsPostTraining = true;            //If true: perform training, and compare parameters after training. Requires getTrainingData() method
    protected boolean testEvaluation = true;                    //If true: perform evaluation. Requires getNewEvaluations() and getEvaluationTestData() methods implemented
    protected boolean testParallelInference = true;             //If true: run the model through ParallelInference. Requires getPredictionsTestData() method. Only applies to DL4J models, NOT SameDiff models
    protected boolean testOverfitting = true;                   //If true: perform overfitting, and ensure the predictions match the training data. Requires both getOverfittingData() and getOverfitNumIterations()

    protected int[] unsupervisedTrainLayersMLN = null;
    protected String[] unsupervisedTrainLayersCG = null;

    //Relative errors for this test case:
    protected double maxRelativeErrorOutput = 1e-4;
    protected double minAbsErrorOutput = 1e-4;
    protected double maxRelativeErrorGradients = 1e-4;
    protected double minAbsErrorGradients = 1e-4;
    protected double maxRelativeErrorPretrainParams = 1e-5;
    protected double minAbsErrorPretrainParams = 1e-5;
    protected double maxRelativeErrorScores = 1e-6;
    protected double minAbsErrorScores = 1e-5;
    protected double maxRelativeErrorParamsPostTraining = 1e-4;
    protected double minAbsErrorParamsPostTraining = 1e-4;
    protected double maxRelativeErrorOverfit = 1e-2;
    protected double minAbsErrorOverfit = 1e-2;

    public abstract ModelType modelType();

    /**
     * Initialize the test case... many tests don't need this; others may use it to download or create data
     * @param testWorkingDir Working directory to use for test
     */
    public void initialize(File testWorkingDir) throws Exception {
        //No op by default
    }

    /**
     * Required if NOT a pretrained model (testType == TestType.RANDOM_INIT)
     */
    public Object getConfiguration() throws Exception {
        throw new RuntimeException("Implementations must override this method if used");
    }

    /**
     * Required for pretrained models (testType == TestType.PRETRAINED)
     */
    public Model getPretrainedModel() throws Exception {
        throw new RuntimeException("Implementations must override this method if used");
    }

    /**
     * Required if testPredictions == true && DL4J model (MultiLayerNetwork or ComputationGraph)
     */
    public List<Pair<INDArray[],INDArray[]>> getPredictionsTestData() throws Exception {
        throw new RuntimeException("Implementations must override this method if used");
    }

    /**
     * Required if testPredictions == true && SameDiff model
     */
    public List<Map<String,INDArray>> getPredictionsTestDataSameDiff() throws Exception {
        throw new RuntimeException("Implementations must override this method if used");
    }

    public List<String> getPredictionsNamesSameDiff() throws Exception {
        throw new RuntimeException("Implementations must override this method if used");
    }

    /**
     * Required if testGradients == true && DL4J model
     */
    public MultiDataSet getGradientsTestData() throws Exception {
        throw new RuntimeException("Implementations must override this method if used");
    }

    /**
     * Required if testGradients == true && SameDiff model
     */
    public Map<String,INDArray> getGradientsTestDataSameDiff() throws Exception {
        throw new RuntimeException("Implementations must override this method if used");
    }

    /**
     * Required when testUnsupervisedTraining == true
     */
    public MultiDataSetIterator getUnsupervisedTrainData() throws Exception {
        throw new RuntimeException("Implementations must override this method if used");
    }

    /**
     * @return Training data - DataSetIterator or MultiDataSetIterator
     */
    public MultiDataSetIterator getTrainingData() throws Exception {
        throw new RuntimeException("Implementations must override this method if used");
    }

    /**
     * Required if testEvaluation == true
     */
    public IEvaluation[] getNewEvaluations() {
        throw new RuntimeException("Implementations must override this method if used");
    }

    public IEvaluation[] doEvaluationSameDiff(SameDiff sd, MultiDataSetIterator iter, IEvaluation[] evaluations){
        throw new RuntimeException("Implementations must override this method if used");
    }

    /**
     * Required if testEvaluation == true
     */
    public MultiDataSetIterator getEvaluationTestData() throws Exception {
        throw new RuntimeException("Implementations must override this method if used");
    }

    /**
     * Required if testOverfitting == true && DL4J model
     */
    public MultiDataSet getOverfittingData() throws Exception {
        throw new RuntimeException("Implementations must override this method if used");
    }

    /**
     * Required if testOverfitting == true && SameDiff model
     */
    public Map<String,INDArray> getOverfittingDataSameDiff() throws Exception {
        throw new RuntimeException("Implementations must override this method if used");
    }

    /**
     * Required if testOverfitting == true
     */
    public int getOverfitNumIterations() {
        throw new RuntimeException("Implementations must override this method if used");
    }


}
