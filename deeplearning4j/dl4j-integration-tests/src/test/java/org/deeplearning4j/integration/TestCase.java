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

package org.deeplearning4j.integration;

import lombok.Data;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.nn.api.Model;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.primitives.Pair;

import java.io.File;
import java.util.List;

/**
 * A single test case for integration tests
 */
@Data
public abstract class TestCase {

    public enum TestType {
        PRETRAINED, RANDOM_INIT
    }

    protected String testName;
    protected TestType testType;
    protected boolean testPredictions = true;
    protected boolean testGradients = true;
    protected boolean testUnsupervisedTraining = false;
    protected boolean testTrainingCurves = true;
    protected boolean testParamsPostTraining = true;
    protected boolean testEvaluation = true;
    protected boolean testParallelInference = true;
    protected boolean testOverfitting = true;

    protected int[] unsupervisedTrainLayersMLN = null;
    protected String[] unsupervisedTrainLayersCG = null;

    //Relative errors for this test case:
    protected double maxRelativeErrorGradients = 1e-6;
    protected double minAbsErrorGradients = 1e-5;
    protected double maxRelativeErrorPretrainParams = 1e-5;
    protected double minAbsErrorPretrainParams = 1e-5;
    protected double maxRelativeErrorScores = 1e-6;
    protected double minAbsErrorScores = 1e-5;
    protected double maxRelativeErrorParamsPostTraining = 1e-4;
    protected double minAbsErrorParamsPostTraining = 1e-4;
    protected double maxRelativeErrorOverfit = 1e-2;
    protected double minAbsErrorOverfit = 1e-2;

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
     * Required if testPredictions == true
     */
    public List<Pair<INDArray[],INDArray[]>> getPredictionsTestData() throws Exception {
        throw new RuntimeException("Implementations must override this method if used");
    }

    /**
     * Required if testGradients == true
     */
    public MultiDataSet getGradientsTestData() throws Exception {
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

    /**
     * Required if testEvaluation == true
     */
    public MultiDataSetIterator getEvaluationTestData() throws Exception {
        throw new RuntimeException("Implementations must override this method if used");
    }

    /**
     * Required if testOverfitting == true
     */
    public MultiDataSet getOverfittingData() throws Exception {
        throw new RuntimeException("Implementations must override this method if used");
    }

    /**
     * Required if testOverfitting == true
     */
    public int getOverfitNumIterations() {
        throw new RuntimeException("Implementations must override this method if used");
    }


}
