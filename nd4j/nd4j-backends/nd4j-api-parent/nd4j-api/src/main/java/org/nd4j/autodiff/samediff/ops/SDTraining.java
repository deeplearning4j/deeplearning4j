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

package org.nd4j.autodiff.samediff.ops;

import static org.nd4j.autodiff.samediff.ops.SDValidation.isSameType;

import java.lang.String;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

public class SDTraining extends SDOps {
  public SDTraining(SameDiff sameDiff) {
    super(sameDiff);
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
   * @return updates Updated gradients (NUMERIC type)
   * @return updatedStateU Updated EMA of squared gradient deviation state (NUMERIC type)
   * @return updatedStateM Updated first moment state (NUMERIC type)
   */
  public SDVariable[] adaBeliefUpdater(SDVariable gradients, SDVariable stateU, SDVariable stateM,
      double lr, double beta1, double beta2, double epsilon, int iteration) {
    SDValidation.validateNumerical("adaBeliefUpdater", "gradients", gradients);
    SDValidation.validateNumerical("adaBeliefUpdater", "stateU", stateU);
    SDValidation.validateNumerical("adaBeliefUpdater", "stateM", stateM);
    return new org.nd4j.linalg.api.ops.impl.updaters.AdaBeliefUpdater(sd,gradients, stateU, stateM, lr, beta1, beta2, epsilon, iteration).outputVariables();
  }

  /**
   * AdaBelief updater.<br>
   * Adapts the step size by the difference between the predicted and observed gradients<br>
   * (the 'belief' in the gradient direction).<br>
   * See: Zhuang et al. (2020) - AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients<br>
   * https://arxiv.org/pdf/2010.07468.pdf<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param gradients Gradients array (NUMERIC type)
   * @param stateU Updater state: exponential moving average of squared gradient deviation (NUMERIC type)
   * @param stateM Updater state: first moment (mean) (NUMERIC type)
   * @param lr Learning rate
   * @param beta1 Beta1 - exponential decay rate for first moment estimate
   * @param beta2 Beta2 - exponential decay rate for second moment estimate
   * @param epsilon Epsilon for numerical stability
   * @param iteration Current iteration (used for bias correction)
   * @return updates Updated gradients (NUMERIC type)
   * @return updatedStateU Updated EMA of squared gradient deviation state (NUMERIC type)
   * @return updatedStateM Updated first moment state (NUMERIC type)
   */
  public SDVariable[] adaBeliefUpdater(String[] names, SDVariable gradients, SDVariable stateU,
      SDVariable stateM, double lr, double beta1, double beta2, double epsilon, int iteration) {
    SDValidation.validateNumerical("adaBeliefUpdater", "gradients", gradients);
    SDValidation.validateNumerical("adaBeliefUpdater", "stateU", stateU);
    SDValidation.validateNumerical("adaBeliefUpdater", "stateM", stateM);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.updaters.AdaBeliefUpdater(sd,gradients, stateU, stateM, lr, beta1, beta2, epsilon, iteration).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
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
   * @return updates Updated gradients (NUMERIC type)
   * @return updatedStateMsg Updated mean squared gradients state (NUMERIC type)
   * @return updatedStateMsdx Updated mean squared delta x state (NUMERIC type)
   */
  public SDVariable[] adaDeltaUpdater(SDVariable gradients, SDVariable stateMsg,
      SDVariable stateMsdx, double rho, double epsilon) {
    SDValidation.validateNumerical("adaDeltaUpdater", "gradients", gradients);
    SDValidation.validateNumerical("adaDeltaUpdater", "stateMsg", stateMsg);
    SDValidation.validateNumerical("adaDeltaUpdater", "stateMsdx", stateMsdx);
    return new org.nd4j.linalg.api.ops.impl.updaters.AdaDeltaUpdater(sd,gradients, stateMsg, stateMsdx, rho, epsilon).outputVariables();
  }

  /**
   * AdaDelta updater.<br>
   * An adaptive learning rate method that uses a moving window of gradient updates to adapt the learning rate.<br>
   * See: Zeiler (2012) - ADADELTA: An Adaptive Learning Rate Method<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param gradients Gradients array (NUMERIC type)
   * @param stateMsg Updater state: mean squared gradients (NUMERIC type)
   * @param stateMsdx Updater state: mean squared delta x (NUMERIC type)
   * @param rho Decay rate (rho)
   * @param epsilon Epsilon for numerical stability
   * @return updates Updated gradients (NUMERIC type)
   * @return updatedStateMsg Updated mean squared gradients state (NUMERIC type)
   * @return updatedStateMsdx Updated mean squared delta x state (NUMERIC type)
   */
  public SDVariable[] adaDeltaUpdater(String[] names, SDVariable gradients, SDVariable stateMsg,
      SDVariable stateMsdx, double rho, double epsilon) {
    SDValidation.validateNumerical("adaDeltaUpdater", "gradients", gradients);
    SDValidation.validateNumerical("adaDeltaUpdater", "stateMsg", stateMsg);
    SDValidation.validateNumerical("adaDeltaUpdater", "stateMsdx", stateMsdx);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.updaters.AdaDeltaUpdater(sd,gradients, stateMsg, stateMsdx, rho, epsilon).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
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
   * @return updates Updated gradients (NUMERIC type)
   * @return updatedState Updated accumulated squared gradients state (NUMERIC type)
   */
  public SDVariable[] adaGradUpdater(SDVariable gradients, SDVariable state, double lr,
      double epsilon) {
    SDValidation.validateNumerical("adaGradUpdater", "gradients", gradients);
    SDValidation.validateNumerical("adaGradUpdater", "state", state);
    return new org.nd4j.linalg.api.ops.impl.updaters.AdaGradUpdater(sd,gradients, state, lr, epsilon).outputVariables();
  }

  /**
   * AdaGrad updater.<br>
   * Adapts the learning rate for each parameter based on accumulated squared gradients.<br>
   * See: Duchi et al. (2011) - Adaptive Subgradient Methods<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param gradients Gradients array (NUMERIC type)
   * @param state Updater state (accumulated squared gradients) (NUMERIC type)
   * @param lr Learning rate
   * @param epsilon Epsilon for numerical stability
   * @return updates Updated gradients (NUMERIC type)
   * @return updatedState Updated accumulated squared gradients state (NUMERIC type)
   */
  public SDVariable[] adaGradUpdater(String[] names, SDVariable gradients, SDVariable state,
      double lr, double epsilon) {
    SDValidation.validateNumerical("adaGradUpdater", "gradients", gradients);
    SDValidation.validateNumerical("adaGradUpdater", "state", state);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.updaters.AdaGradUpdater(sd,gradients, state, lr, epsilon).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
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
   * @return updates Updated gradients (NUMERIC type)
   * @return updatedStateU Updated infinity norm state (NUMERIC type)
   * @return updatedStateM Updated first moment state (NUMERIC type)
   */
  public SDVariable[] adaMaxUpdater(SDVariable gradients, SDVariable stateU, SDVariable stateM,
      double lr, double beta1, double beta2, double epsilon, int iteration) {
    SDValidation.validateNumerical("adaMaxUpdater", "gradients", gradients);
    SDValidation.validateNumerical("adaMaxUpdater", "stateU", stateU);
    SDValidation.validateNumerical("adaMaxUpdater", "stateM", stateM);
    return new org.nd4j.linalg.api.ops.impl.updaters.AdaMaxUpdater(sd,gradients, stateU, stateM, lr, beta1, beta2, epsilon, iteration).outputVariables();
  }

  /**
   * AdaMax updater.<br>
   * A variant of Adam based on the infinity norm, which can be more stable in some settings.<br>
   * See: Kingma and Ba (2014) - Adam: A Method for Stochastic Optimization (Section 7.1)<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param gradients Gradients array (NUMERIC type)
   * @param stateU Updater state: infinity norm (max) (NUMERIC type)
   * @param stateM Updater state: first moment (mean) (NUMERIC type)
   * @param lr Learning rate
   * @param beta1 Beta1 - exponential decay rate for first moment estimate
   * @param beta2 Beta2 - exponential decay rate for infinity norm
   * @param epsilon Epsilon for numerical stability
   * @param iteration Current iteration (used for bias correction)
   * @return updates Updated gradients (NUMERIC type)
   * @return updatedStateU Updated infinity norm state (NUMERIC type)
   * @return updatedStateM Updated first moment state (NUMERIC type)
   */
  public SDVariable[] adaMaxUpdater(String[] names, SDVariable gradients, SDVariable stateU,
      SDVariable stateM, double lr, double beta1, double beta2, double epsilon, int iteration) {
    SDValidation.validateNumerical("adaMaxUpdater", "gradients", gradients);
    SDValidation.validateNumerical("adaMaxUpdater", "stateU", stateU);
    SDValidation.validateNumerical("adaMaxUpdater", "stateM", stateM);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.updaters.AdaMaxUpdater(sd,gradients, stateU, stateM, lr, beta1, beta2, epsilon, iteration).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
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
   * @return updates Updated gradients (NUMERIC type)
   * @return updatedStateU Updated first moment state (NUMERIC type)
   * @return updatedStateM Updated second moment state (NUMERIC type)
   */
  public SDVariable[] adamUpdater(SDVariable gradients, SDVariable stateU, SDVariable stateM,
      double lr, double beta1, double beta2, double epsilon, int iteration) {
    SDValidation.validateNumerical("adamUpdater", "gradients", gradients);
    SDValidation.validateNumerical("adamUpdater", "stateU", stateU);
    SDValidation.validateNumerical("adamUpdater", "stateM", stateM);
    return new org.nd4j.linalg.api.ops.impl.updaters.AdamUpdater(sd,gradients, stateU, stateM, lr, beta1, beta2, epsilon, iteration).outputVariables();
  }

  /**
   * Adam updater.<br>
   * Adaptive Moment Estimation - computes adaptive learning rates for each parameter.<br>
   * See: Kingma and Ba (2014) - Adam: A Method for Stochastic Optimization<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param gradients Gradients array (NUMERIC type)
   * @param stateU Updater state: first moment (mean) (NUMERIC type)
   * @param stateM Updater state: second moment (uncentered variance) (NUMERIC type)
   * @param lr Learning rate
   * @param beta1 Beta1 - exponential decay rate for first moment estimate
   * @param beta2 Beta2 - exponential decay rate for second moment estimate
   * @param epsilon Epsilon for numerical stability
   * @param iteration Current iteration (used for bias correction)
   * @return updates Updated gradients (NUMERIC type)
   * @return updatedStateU Updated first moment state (NUMERIC type)
   * @return updatedStateM Updated second moment state (NUMERIC type)
   */
  public SDVariable[] adamUpdater(String[] names, SDVariable gradients, SDVariable stateU,
      SDVariable stateM, double lr, double beta1, double beta2, double epsilon, int iteration) {
    SDValidation.validateNumerical("adamUpdater", "gradients", gradients);
    SDValidation.validateNumerical("adamUpdater", "stateU", stateU);
    SDValidation.validateNumerical("adamUpdater", "stateM", stateM);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.updaters.AdamUpdater(sd,gradients, stateU, stateM, lr, beta1, beta2, epsilon, iteration).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
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
   * @return updates Updated gradients (NUMERIC type)
   * @return updatedStateV Updated second moment state (NUMERIC type)
   * @return updatedStateM Updated first moment state (NUMERIC type)
   * @return updatedStateH Updated maximum second moment state (NUMERIC type)
   */
  public SDVariable[] amsGradUpdater(SDVariable gradients, SDVariable stateV, SDVariable stateM,
      SDVariable stateH, double lr, double beta1, double beta2, double epsilon, int iteration) {
    SDValidation.validateNumerical("amsGradUpdater", "gradients", gradients);
    SDValidation.validateNumerical("amsGradUpdater", "stateV", stateV);
    SDValidation.validateNumerical("amsGradUpdater", "stateM", stateM);
    SDValidation.validateNumerical("amsGradUpdater", "stateH", stateH);
    return new org.nd4j.linalg.api.ops.impl.updaters.AmsGradUpdater(sd,gradients, stateV, stateM, stateH, lr, beta1, beta2, epsilon, iteration).outputVariables();
  }

  /**
   * AMSGrad updater.<br>
   * A variant of Adam that uses the maximum of past squared gradients to provide<br>
   * better convergence guarantees.<br>
   * See: Reddi et al. (2018) - On the Convergence of Adam and Beyond<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param gradients Gradients array (NUMERIC type)
   * @param stateV Updater state: second moment (uncentered variance) (NUMERIC type)
   * @param stateM Updater state: first moment (mean) (NUMERIC type)
   * @param stateH Updater state: maximum of past second moments (NUMERIC type)
   * @param lr Learning rate
   * @param beta1 Beta1 - exponential decay rate for first moment estimate
   * @param beta2 Beta2 - exponential decay rate for second moment estimate
   * @param epsilon Epsilon for numerical stability
   * @param iteration Current iteration (used for bias correction)
   * @return updates Updated gradients (NUMERIC type)
   * @return updatedStateV Updated second moment state (NUMERIC type)
   * @return updatedStateM Updated first moment state (NUMERIC type)
   * @return updatedStateH Updated maximum second moment state (NUMERIC type)
   */
  public SDVariable[] amsGradUpdater(String[] names, SDVariable gradients, SDVariable stateV,
      SDVariable stateM, SDVariable stateH, double lr, double beta1, double beta2, double epsilon,
      int iteration) {
    SDValidation.validateNumerical("amsGradUpdater", "gradients", gradients);
    SDValidation.validateNumerical("amsGradUpdater", "stateV", stateV);
    SDValidation.validateNumerical("amsGradUpdater", "stateM", stateM);
    SDValidation.validateNumerical("amsGradUpdater", "stateH", stateH);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.updaters.AmsGradUpdater(sd,gradients, stateV, stateM, stateH, lr, beta1, beta2, epsilon, iteration).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
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
   * @return updates Updated gradients (NUMERIC type)
   * @return updatedStateV Updated second moment state (NUMERIC type)
   * @return updatedStateM Updated first moment state (NUMERIC type)
   */
  public SDVariable[] nadamUpdater(SDVariable gradients, SDVariable stateV, SDVariable stateM,
      double lr, double beta1, double beta2, double epsilon, int iteration) {
    SDValidation.validateNumerical("nadamUpdater", "gradients", gradients);
    SDValidation.validateNumerical("nadamUpdater", "stateV", stateV);
    SDValidation.validateNumerical("nadamUpdater", "stateM", stateM);
    return new org.nd4j.linalg.api.ops.impl.updaters.NadamUpdater(sd,gradients, stateV, stateM, lr, beta1, beta2, epsilon, iteration).outputVariables();
  }

  /**
   * Nadam updater.<br>
   * Adam with Nesterov momentum. Combines Adam and Nesterov accelerated gradient.<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param gradients Gradients array (NUMERIC type)
   * @param stateV Updater state: second moment (uncentered variance) (NUMERIC type)
   * @param stateM Updater state: first moment (mean) (NUMERIC type)
   * @param lr Learning rate
   * @param beta1 Beta1 - exponential decay rate for first moment estimate
   * @param beta2 Beta2 - exponential decay rate for second moment estimate
   * @param epsilon Epsilon for numerical stability
   * @param iteration Current iteration (used for bias correction)
   * @return updates Updated gradients (NUMERIC type)
   * @return updatedStateV Updated second moment state (NUMERIC type)
   * @return updatedStateM Updated first moment state (NUMERIC type)
   */
  public SDVariable[] nadamUpdater(String[] names, SDVariable gradients, SDVariable stateV,
      SDVariable stateM, double lr, double beta1, double beta2, double epsilon, int iteration) {
    SDValidation.validateNumerical("nadamUpdater", "gradients", gradients);
    SDValidation.validateNumerical("nadamUpdater", "stateV", stateV);
    SDValidation.validateNumerical("nadamUpdater", "stateM", stateM);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.updaters.NadamUpdater(sd,gradients, stateV, stateM, lr, beta1, beta2, epsilon, iteration).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
  }

  /**
   * Nesterov momentum SGD updater.<br>
   * Applies Nesterov accelerated gradient update using momentum.<br>
   *
   * @param gradients Gradients array (NUMERIC type)
   * @param state Updater state (velocity) (NUMERIC type)
   * @param lr Learning rate
   * @param momentum Momentum coefficient
   * @return updates Updated gradients (NUMERIC type)
   * @return updatedState Updated velocity state (NUMERIC type)
   */
  public SDVariable[] nesterovsUpdater(SDVariable gradients, SDVariable state, double lr,
      double momentum) {
    SDValidation.validateNumerical("nesterovsUpdater", "gradients", gradients);
    SDValidation.validateNumerical("nesterovsUpdater", "state", state);
    return new org.nd4j.linalg.api.ops.impl.updaters.NesterovsUpdater(sd,gradients, state, lr, momentum).outputVariables();
  }

  /**
   * Nesterov momentum SGD updater.<br>
   * Applies Nesterov accelerated gradient update using momentum.<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param gradients Gradients array (NUMERIC type)
   * @param state Updater state (velocity) (NUMERIC type)
   * @param lr Learning rate
   * @param momentum Momentum coefficient
   * @return updates Updated gradients (NUMERIC type)
   * @return updatedState Updated velocity state (NUMERIC type)
   */
  public SDVariable[] nesterovsUpdater(String[] names, SDVariable gradients, SDVariable state,
      double lr, double momentum) {
    SDValidation.validateNumerical("nesterovsUpdater", "gradients", gradients);
    SDValidation.validateNumerical("nesterovsUpdater", "state", state);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.updaters.NesterovsUpdater(sd,gradients, state, lr, momentum).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
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
   * @return updates Updated gradients (NUMERIC type)
   * @return updatedState Updated mean squared gradients state (NUMERIC type)
   */
  public SDVariable[] rmsPropUpdater(SDVariable gradients, SDVariable state, double lr,
      double decay, double epsilon) {
    SDValidation.validateNumerical("rmsPropUpdater", "gradients", gradients);
    SDValidation.validateNumerical("rmsPropUpdater", "state", state);
    return new org.nd4j.linalg.api.ops.impl.updaters.RmsPropUpdater(sd,gradients, state, lr, decay, epsilon).outputVariables();
  }

  /**
   * RMSProp updater.<br>
   * Divides the learning rate by a running average of the magnitudes of recent gradients.<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param gradients Gradients array (NUMERIC type)
   * @param state Updater state (mean squared gradients) (NUMERIC type)
   * @param lr Learning rate
   * @param decay Decay rate (rho)
   * @param epsilon Epsilon for numerical stability
   * @return updates Updated gradients (NUMERIC type)
   * @return updatedState Updated mean squared gradients state (NUMERIC type)
   */
  public SDVariable[] rmsPropUpdater(String[] names, SDVariable gradients, SDVariable state,
      double lr, double decay, double epsilon) {
    SDValidation.validateNumerical("rmsPropUpdater", "gradients", gradients);
    SDValidation.validateNumerical("rmsPropUpdater", "state", state);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.updaters.RmsPropUpdater(sd,gradients, state, lr, decay, epsilon).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
  }

  /**
   * SGD (Stochastic Gradient Descent) updater.<br>
   * Applies gradient update: param -= lr * gradient<br>
   *
   * @param gradients Gradients array (NUMERIC type)
   * @param lr Learning rate
   * @return output Updated gradients (in-place) (NUMERIC type)
   */
  public SDVariable sgdUpdater(SDVariable gradients, double lr) {
    SDValidation.validateNumerical("sgdUpdater", "gradients", gradients);
    return new org.nd4j.linalg.api.ops.impl.updaters.SgdUpdater(sd,gradients, lr).outputVariable();
  }

  /**
   * SGD (Stochastic Gradient Descent) updater.<br>
   * Applies gradient update: param -= lr * gradient<br>
   *
   * @param name name May be null. Name for the output variable
   * @param gradients Gradients array (NUMERIC type)
   * @param lr Learning rate
   * @return output Updated gradients (in-place) (NUMERIC type)
   */
  public SDVariable sgdUpdater(String name, SDVariable gradients, double lr) {
    SDValidation.validateNumerical("sgdUpdater", "gradients", gradients);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.updaters.SgdUpdater(sd,gradients, lr).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }
}
