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

package org.arbiter.optimize.solvers;

import java.util.Collection;
import java.util.LinkedList;
import org.arbiter.berkeley.Pair;
import org.arbiter.nn.api.Model;
import org.arbiter.nn.conf.BaseNeuralNetConfiguration;
import org.arbiter.nn.gradient.Gradient;
import org.arbiter.optimize.api.IterationListener;
import org.arbiter.optimize.api.StepFunction;
import org.arbiter.optimize.api.TerminationCondition;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * LBFGS
 * @author Adam Gibson
 */
public class LBFGS extends BaseOptimizer {

    private int m = 4;


    public LBFGS(BaseNeuralNetConfiguration conf, StepFunction stepFunction,
                 Collection<IterationListener> iterationListeners, Model model) {
        super(conf, stepFunction, iterationListeners, model);
    }

    public LBFGS(BaseNeuralNetConfiguration conf, StepFunction stepFunction,
                 Collection<IterationListener> iterationListeners,
                 Collection<TerminationCondition> terminationConditions, Model model) {
        super(conf, stepFunction, iterationListeners, terminationConditions, model);
    }

    @Override
    protected boolean preFirstStepProcess(INDArray gradient) {
        //initial direction should be normal
        searchState.put(GRADIENT_KEY,gradient.mul(Nd4j.norm2(gradient).rdivi(1.0).getDouble(0)));
        return true;
    }

    @Override
    public void setupSearchState(Pair<Gradient, Double> pair) {
        super.setupSearchState(pair);
        INDArray gradient = (INDArray) searchState.get(GRADIENT_KEY);
        INDArray params = (INDArray) searchState.get(PARAMS_KEY);
        searchState.put("s",new LinkedList());
        searchState.put("y",new LinkedList());
        searchState.put("rho",new LinkedList());
        searchState.put("alpha", Nd4j.create(m));
        searchState.put("oldparams",params.dup());
        searchState.put("oldgradient",gradient.dup());


    }

    @Override
    protected void postFirstStep(INDArray gradient) {
        super.postFirstStep(gradient);
        if(step == 0.0) {
            log.info("Unable to step in that direction...resetting");
            setupSearchState(model.gradientAndScore());
            step = 1.0;
        }

    }

    @Override
    public void preProcessLine(INDArray line) {
        INDArray oldParameters = (INDArray) searchState.get("oldparams");
        INDArray params = (INDArray) searchState.get(PARAMS_KEY);
        oldParameters.assign(params.sub(oldParameters));
        INDArray oldGradient = (INDArray) searchState.get("oldgradient");
        INDArray gradient = (INDArray) searchState.get(GRADIENT_KEY);
        oldGradient.subi(gradient);

        double sy = Nd4j.getBlasWrapper().dot(oldParameters,oldGradient) + Nd4j.EPS_THRESHOLD;
        double yy = Transforms.pow(oldGradient,2).sum(Integer.MAX_VALUE).getDouble(0) + Nd4j.EPS_THRESHOLD;
        double gamma = sy / yy;

        LinkedList<Double> rho = (LinkedList<Double>) searchState.get("rho");
        rho.add(1.0 / sy);

        LinkedList<INDArray> s = (LinkedList<INDArray>) searchState.get("s");
        s.add(oldParameters);

        LinkedList<INDArray> y = (LinkedList<INDArray>) searchState.get("y");
        y.add(oldGradient);

        if(s.size() != y.size())
            throw new IllegalStateException("S and y mis matched sizes");

        INDArray alpha = (INDArray) searchState.get("alpha");
        // First work backwards, from the most recent difference vectors
        for (int i = s.size() - 1; i >= 0; i--) {
            if(s.get(i).length() != gradient.length())
                throw new IllegalStateException("Gradient and s length not equal");
            if(i >= alpha.length())
                break;
            if(i > rho.size())
                throw new IllegalStateException("I > rho size");
            alpha.putScalar(i, rho.get(i) * Nd4j.getBlasWrapper().dot(gradient, s.get(i)));
            if(alpha.data().dataType() == DataBuffer.DOUBLE)
                Nd4j.getBlasWrapper().axpy(-1.0 * alpha.getDouble(i), gradient, y.get(i));
            else
                Nd4j.getBlasWrapper().axpy(-1.0f * alpha.getFloat(i), gradient, y.get(i));

        }


        gradient.muli(gamma);

        // Now work forwards, from the oldest to the newest difference vectors
        for (int i = 0; i < y.size(); i++) {
            if(i >= alpha.length())
                break;
            double beta = rho.get(i) * Nd4j.getBlasWrapper().dot(y.get(i),gradient);
            if(alpha.data().dataType() == DataBuffer.DOUBLE)
                Nd4j.getBlasWrapper().axpy(alpha.getDouble(i) * beta, gradient, s.get(i));
            else
                Nd4j.getBlasWrapper().axpy(alpha.getFloat(i) * (float) beta, gradient, s.get(i));

        }

        oldParameters.assign(params);
        oldGradient.assign(gradient);
        gradient.muli(-1);


    }

    @Override
    public void postStep() {

    }
}
