/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.evaluation;

import static org.junit.Assert.assertEquals;

import org.junit.Test;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.EvaluationBinary;
import org.nd4j.evaluation.classification.EvaluationCalibration;
import org.nd4j.evaluation.classification.ROC;
import org.nd4j.evaluation.classification.ROCBinary;
import org.nd4j.evaluation.classification.ROCMultiClass;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

public class NewInstanceTest extends BaseNd4jTest {

    public NewInstanceTest(Nd4jBackend backend) {
        super(backend);
    }

    @Override
    public char ordering() {
        return 'c';
    }

    @Test
    public void testNewInstances() {
        boolean print = true;
        Nd4j.getRandom().setSeed(12345);

        Evaluation evaluation = new Evaluation();
        EvaluationBinary evaluationBinary = new EvaluationBinary();
        ROC roc = new ROC(2);
        ROCBinary roc2 = new ROCBinary(2);
        ROCMultiClass roc3 = new ROCMultiClass(2);
        RegressionEvaluation regressionEvaluation = new RegressionEvaluation();
        EvaluationCalibration ec = new EvaluationCalibration();


        IEvaluation[] arr = new IEvaluation[] {evaluation, evaluationBinary, roc, roc2, roc3, regressionEvaluation, ec};

        INDArray evalLabel1 = Nd4j.create(10, 3);
        for (int i = 0; i < 10; i++) {
            evalLabel1.putScalar(i, i % 3, 1.0);
        }
        INDArray evalProb1 = Nd4j.rand(10, 3);
        evalProb1.diviColumnVector(evalProb1.sum(1));

        evaluation.eval(evalLabel1, evalProb1);
        roc3.eval(evalLabel1, evalProb1);
        ec.eval(evalLabel1, evalProb1);

        INDArray evalLabel2 = Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.createUninitialized(10, 3), 0.5));
        INDArray evalProb2 = Nd4j.rand(10, 3);
        evaluationBinary.eval(evalLabel2, evalProb2);
        roc2.eval(evalLabel2, evalProb2);

        INDArray evalLabel3 = Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.createUninitialized(10, 1), 0.5));
        INDArray evalProb3 = Nd4j.rand(10, 1);
        roc.eval(evalLabel3, evalProb3);

        INDArray reg1 = Nd4j.rand(10, 3);
        INDArray reg2 = Nd4j.rand(10, 3);

        regressionEvaluation.eval(reg1, reg2);

        Evaluation evaluation2 = evaluation.newInstance();
        EvaluationBinary evaluationBinary2 = evaluationBinary.newInstance();
        ROC roc_2 = roc.newInstance();
        ROCBinary roc22 = roc2.newInstance();
        ROCMultiClass roc32 = roc3.newInstance();
        RegressionEvaluation regressionEvaluation2 = regressionEvaluation.newInstance();
        EvaluationCalibration ec2 = ec.newInstance();

        IEvaluation[] arr2 = new IEvaluation[] {evaluation2, evaluationBinary2, roc_2, roc22, roc32, regressionEvaluation2, ec2};

        evaluation2.eval(evalLabel1, evalProb1);
        roc32.eval(evalLabel1, evalProb1);
        ec2.eval(evalLabel1, evalProb1);

        evaluationBinary2.eval(evalLabel2, evalProb2);
        roc22.eval(evalLabel2, evalProb2);

        roc_2.eval(evalLabel3, evalProb3);

        regressionEvaluation2.eval(reg1, reg2);

        for (int i = 0 ; i < arr.length ; i++) {

            IEvaluation e = arr[i];
            IEvaluation e2 = arr2[i];
            assertEquals("Json not equal ", e.toJson(), e2.toJson());
            assertEquals("Stats not equal ", e.stats(), e2.stats());
        }
    }

}
