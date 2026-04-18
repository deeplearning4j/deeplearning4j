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

//================== GENERATED CODE - DO NOT MODIFY THIS FILE ==================

package org.nd4j.linalg.factory.ops;

import static org.nd4j.linalg.factory.NDValidation.isSameType;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.NDValidation;
import org.nd4j.linalg.factory.Nd4j;

public class NDTraining {
  public NDTraining() {
  }

  /**
   * SGD (Stochastic Gradient Descent) updater.<br>
   * Applies gradient update: param -= lr * gradient<br>
   *
   * @param gradients Gradients array (NUMERIC type)
   * @param lr Learning rate
   * @return output Updated gradients (in-place) (NUMERIC type)
   */
  public INDArray sgdUpdater(INDArray gradients, double lr) {
    NDValidation.validateNumerical("sgdUpdater", "gradients", gradients);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.updaters.SgdUpdater(gradients, lr));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Nesterov momentum SGD updater.<br>
   * Applies Nesterov accelerated gradient update using momentum.<br>
   *
   * @param gradients Gradients array (NUMERIC type)
   * @param state Updater state (velocity) (NUMERIC type)
   * @param lr Learning rate
   * @param momentum Momentum coefficient
   * @return output [updates, updatedState] (NUMERIC type)
   */
  public INDArray[] nesterovsUpdater(INDArray gradients, INDArray state, double lr,
      double momentum) {
    NDValidation.validateNumerical("nesterovsUpdater", "gradients", gradients);
    NDValidation.validateNumerical("nesterovsUpdater", "state", state);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.updaters.NesterovsUpdater(gradients, state, lr, momentum));
  }

  /**
   * AdaGrad updater.<br>
   * Adapts the learning rate for each parameter based on accumulated squared gradients.<br>
   * See: Duchi et al. (2011) - Adaptive Subgradient Methods<br>
   *
   * @param gradients Gradients array (NUMERIC type)
   * @param state Updater state (accumulated squared gradients) (NUMERIC type)
   * @param lr Learning rate
   * @param epsilon Epsilon for numerical stability
   * @return output [updates, updatedState] (NUMERIC type)
   */
  public INDArray[] adaGradUpdater(INDArray gradients, INDArray state, double lr, double epsilon) {
    NDValidation.validateNumerical("adaGradUpdater", "gradients", gradients);
    NDValidation.validateNumerical("adaGradUpdater", "state", state);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.updaters.AdaGradUpdater(gradients, state, lr, epsilon));
  }

  /**
   * AdaDelta updater.<br>
   * An adaptive learning rate method that uses a moving window of gradient updates to adapt the learning rate.<br>
   * See: Zeiler (2012) - ADADELTA: An Adaptive Learning Rate Method<br>
   *
   * @param gradients Gradients array (NUMERIC type)
   * @param stateMsg Updater state: mean squared gradients (NUMERIC type)
   * @param stateMsdx Updater state: mean squared delta x (NUMERIC type)
   * @param rho Decay rate (rho)
   * @param epsilon Epsilon for numerical stability
   * @return output [updates, updatedStateMsg, updatedStateMsdx] (NUMERIC type)
   */
  public INDArray[] adaDeltaUpdater(INDArray gradients, INDArray stateMsg, INDArray stateMsdx,
      double rho, double epsilon) {
    NDValidation.validateNumerical("adaDeltaUpdater", "gradients", gradients);
    NDValidation.validateNumerical("adaDeltaUpdater", "stateMsg", stateMsg);
    NDValidation.validateNumerical("adaDeltaUpdater", "stateMsdx", stateMsdx);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.updaters.AdaDeltaUpdater(gradients, stateMsg, stateMsdx, rho, epsilon));
  }

  /**
   * RMSProp updater.<br>
   * Divides the learning rate by a running average of the magnitudes of recent gradients.<br>
   *
   * @param gradients Gradients array (NUMERIC type)
   * @param state Updater state (mean squared gradients) (NUMERIC type)
   * @param lr Learning rate
   * @param decay Decay rate (rho)
   * @param epsilon Epsilon for numerical stability
   * @return output [updates, updatedState] (NUMERIC type)
   */
  public INDArray[] rmsPropUpdater(INDArray gradients, INDArray state, double lr, double decay,
      double epsilon) {
    NDValidation.validateNumerical("rmsPropUpdater", "gradients", gradients);
    NDValidation.validateNumerical("rmsPropUpdater", "state", state);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.updaters.RmsPropUpdater(gradients, state, lr, decay, epsilon));
  }

  /**
   * Adam updater.<br>
   * Adaptive Moment Estimation - computes adaptive learning rates for each parameter.<br>
   * See: Kingma and Ba (2014) - Adam: A Method for Stochastic Optimization<br>
   *
   * @param gradients Gradients array (NUMERIC type)
   * @param stateU Updater state: first moment (mean) (NUMERIC type)
   * @param stateM Updater state: second moment (uncentered variance) (NUMERIC type)
   * @param lr Learning rate
   * @param beta1 Beta1 - exponential decay rate for first moment estimate
   * @param beta2 Beta2 - exponential decay rate for second moment estimate
   * @param epsilon Epsilon for numerical stability
   * @param iteration Current iteration (used for bias correction)
   * @return output [updates, updatedStateU, updatedStateM] (NUMERIC type)
   */
  public INDArray[] adamUpdater(INDArray gradients, INDArray stateU, INDArray stateM, double lr,
      double beta1, double beta2, double epsilon, int iteration) {
    NDValidation.validateNumerical("adamUpdater", "gradients", gradients);
    NDValidation.validateNumerical("adamUpdater", "stateU", stateU);
    NDValidation.validateNumerical("adamUpdater", "stateM", stateM);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.updaters.AdamUpdater(gradients, stateU, stateM, lr, beta1, beta2, epsilon, iteration));
  }

  /**
   * Nadam updater.<br>
   * Adam with Nesterov momentum. Combines Adam and Nesterov accelerated gradient.<br>
   *
   * @param gradients Gradients array (NUMERIC type)
   * @param stateV Updater state: second moment (uncentered variance) (NUMERIC type)
   * @param stateM Updater state: first moment (mean) (NUMERIC type)
   * @param lr Learning rate
   * @param beta1 Beta1 - exponential decay rate for first moment estimate
   * @param beta2 Beta2 - exponential decay rate for second moment estimate
   * @param epsilon Epsilon for numerical stability
   * @param iteration Current iteration (used for bias correction)
   * @return output [updates, updatedStateV, updatedStateM] (NUMERIC type)
   */
  public INDArray[] nadamUpdater(INDArray gradients, INDArray stateV, INDArray stateM, double lr,
      double beta1, double beta2, double epsilon, int iteration) {
    NDValidation.validateNumerical("nadamUpdater", "gradients", gradients);
    NDValidation.validateNumerical("nadamUpdater", "stateV", stateV);
    NDValidation.validateNumerical("nadamUpdater", "stateM", stateM);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.updaters.NadamUpdater(gradients, stateV, stateM, lr, beta1, beta2, epsilon, iteration));
  }

  /**
   * AdaMax updater.<br>
   * A variant of Adam based on the infinity norm, which can be more stable in some settings.<br>
   * See: Kingma and Ba (2014) - Adam: A Method for Stochastic Optimization (Section 7.1)<br>
   *
   * @param gradients Gradients array (NUMERIC type)
   * @param stateU Updater state: infinity norm (max) (NUMERIC type)
   * @param stateM Updater state: first moment (mean) (NUMERIC type)
   * @param lr Learning rate
   * @param beta1 Beta1 - exponential decay rate for first moment estimate
   * @param beta2 Beta2 - exponential decay rate for infinity norm
   * @param epsilon Epsilon for numerical stability
   * @param iteration Current iteration (used for bias correction)
   * @return output [updates, updatedStateU, updatedStateM] (NUMERIC type)
   */
  public INDArray[] adaMaxUpdater(INDArray gradients, INDArray stateU, INDArray stateM, double lr,
      double beta1, double beta2, double epsilon, int iteration) {
    NDValidation.validateNumerical("adaMaxUpdater", "gradients", gradients);
    NDValidation.validateNumerical("adaMaxUpdater", "stateU", stateU);
    NDValidation.validateNumerical("adaMaxUpdater", "stateM", stateM);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.updaters.AdaMaxUpdater(gradients, stateU, stateM, lr, beta1, beta2, epsilon, iteration));
  }

  /**
   * AMSGrad updater.<br>
   * A variant of Adam that uses the maximum of past squared gradients to provide<br>
   * better convergence guarantees.<br>
   * See: Reddi et al. (2018) - On the Convergence of Adam and Beyond<br>
   *
   * @param gradients Gradients array (NUMERIC type)
   * @param stateV Updater state: second moment (uncentered variance) (NUMERIC type)
   * @param stateM Updater state: first moment (mean) (NUMERIC type)
   * @param stateH Updater state: maximum of past second moments (NUMERIC type)
   * @param lr Learning rate
   * @param beta1 Beta1 - exponential decay rate for first moment estimate
   * @param beta2 Beta2 - exponential decay rate for second moment estimate
   * @param epsilon Epsilon for numerical stability
   * @param iteration Current iteration (used for bias correction)
   * @return output [updates, updatedStateV, updatedStateM, updatedStateH] (NUMERIC type)
   */
  public INDArray[] amsGradUpdater(INDArray gradients, INDArray stateV, INDArray stateM,
      INDArray stateH, double lr, double beta1, double beta2, double epsilon, int iteration) {
    NDValidation.validateNumerical("amsGradUpdater", "gradients", gradients);
    NDValidation.validateNumerical("amsGradUpdater", "stateV", stateV);
    NDValidation.validateNumerical("amsGradUpdater", "stateM", stateM);
    NDValidation.validateNumerical("amsGradUpdater", "stateH", stateH);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.updaters.AmsGradUpdater(gradients, stateV, stateM, stateH, lr, beta1, beta2, epsilon, iteration));
  }

  /**
   * AdaBelief updater.<br>
   * Adapts the step size by the difference between the predicted and observed gradients<br>
   * (the 'belief' in the gradient direction).<br>
   * See: Zhuang et al. (2020) - AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients<br>
   * https://arxiv.org/pdf/2010.07468.pdf<br>
   *
   * @param gradients Gradients array (NUMERIC type)
   * @param stateU Updater state: exponential moving average of squared gradient deviation (NUMERIC type)
   * @param stateM Updater state: first moment (mean) (NUMERIC type)
   * @param lr Learning rate
   * @param beta1 Beta1 - exponential decay rate for first moment estimate
   * @param beta2 Beta2 - exponential decay rate for second moment estimate
   * @param epsilon Epsilon for numerical stability
   * @param iteration Current iteration (used for bias correction)
   * @return output [updates, updatedStateU, updatedStateM] (NUMERIC type)
   */
  public INDArray[] adaBeliefUpdater(INDArray gradients, INDArray stateU, INDArray stateM,
      double lr, double beta1, double beta2, double epsilon, int iteration) {
    NDValidation.validateNumerical("adaBeliefUpdater", "gradients", gradients);
    NDValidation.validateNumerical("adaBeliefUpdater", "stateU", stateU);
    NDValidation.validateNumerical("adaBeliefUpdater", "stateM", stateM);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.updaters.AdaBeliefUpdater(gradients, stateU, stateM, lr, beta1, beta2, epsilon, iteration));
  }
}
