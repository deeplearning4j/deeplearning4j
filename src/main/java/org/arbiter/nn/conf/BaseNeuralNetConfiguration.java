/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.arbiter.nn.conf;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.arbiter.nn.api.OptimizationAlgorithm;
import org.arbiter.optimize.api.IterationListener;
import org.arbiter.optimize.api.StepFunction;
import org.arbiter.optimize.stepfunctions.GradientStepFunction;

public abstract class BaseNeuralNetConfiguration {

  //number of line search iterations
  protected int numLineSearchIterations = 100;

  protected int numIterations = 1000;

  //gradient keys used for ensuring order when getting and setting the gradient
  protected List<String> variables = new ArrayList<>();

  /* momentum for learning */
  protected double momentum = 0.5;

  //reset adagrad historical gradient after n iterations
  protected int resetAdaGradIterations = -1;

  //momentum after n iterations
  protected Map<Integer,Double> momentumAfter = new HashMap<>();

  protected boolean useRegularization = false;

  protected boolean useAdaGrad = true;

  /* L2 Regularization constant */
  protected double l2 = 0;

  //whether to constrain the gradient to unit norm or not
  protected boolean constrainGradientToUnitNorm = false;

  private double lr = 1e-1;

  protected transient List<IterationListener> listeners;

  protected transient StepFunction stepFunction = new GradientStepFunction();

  protected OptimizationAlgorithm optimizationAlgo = OptimizationAlgorithm.CONJUGATE_GRADIENT;

  public double getL2() {
    return l2;
  }

  public void setL2(double l2) {
    this.l2 = l2;
  }

  public int getNumLineSearchIterations() {
    return numLineSearchIterations;
  }

  public void setNumLineSearchIterations(int numLineSearchIterations) {
    this.numLineSearchIterations = numLineSearchIterations;
  }

  public int getNumIterations() {
    return numIterations;
  }

  public void setNumIterations(int numIterations) {
    this.numIterations = numIterations;
  }

  public List<String> getVariables() {
    return variables;
  }

  public void setVariables(List<String> variables) {
    this.variables = variables;
  }

  public double getMomentum() {
    return momentum;
  }

  public void setMomentum(double momentum) {
    this.momentum = momentum;
  }

  public int getResetAdaGradIterations() {
    return resetAdaGradIterations;
  }

  public void setResetAdaGradIterations(int resetAdaGradIterations) {
    this.resetAdaGradIterations = resetAdaGradIterations;
  }

  public Map<Integer, Double> getMomentumAfter() {
    return momentumAfter;
  }

  public void setMomentumAfter(Map<Integer, Double> momentumAfter) {
    this.momentumAfter = momentumAfter;
  }

  public boolean isUseRegularization() {
    return useRegularization;
  }

  public void setUseRegularization(boolean useRegularization) {
    this.useRegularization = useRegularization;
  }

  public boolean isUseAdaGrad() {
    return useAdaGrad;
  }

  public void setUseAdaGrad(boolean useAdaGrad) {
    this.useAdaGrad = useAdaGrad;
  }

  public boolean isConstrainGradientToUnitNorm() {
    return constrainGradientToUnitNorm;
  }

  public void setConstrainGradientToUnitNorm(boolean constrainGradientToUnitNorm) {
    this.constrainGradientToUnitNorm = constrainGradientToUnitNorm;
  }

  public double getLr() {
    return lr;
  }

  public void setLr(double lr) {
    this.lr = lr;
  }

  public OptimizationAlgorithm getOptimizationAlgo() {
    return optimizationAlgo;
  }

  public void setOptimizationAlgo(OptimizationAlgorithm optimizationAlgo) {
    this.optimizationAlgo = optimizationAlgo;
  }

  public List<IterationListener> getListeners() {
    return listeners;
  }

  public void setListeners(List<IterationListener> listeners) {
    this.listeners = listeners;
  }

  public StepFunction getStepFunction() {
    return stepFunction;
  }

  public void setStepFunction(StepFunction stepFunction) {
    this.stepFunction = stepFunction;
  }
}
