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

package org.deeplearning4j.nn.layers.normalization;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.deeplearning4j.common.config.DL4JClassLoading;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.CNN2DFormat;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.deeplearning4j.nn.layers.LayerHelper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.MulOp;
import org.nd4j.linalg.exception.ND4JOpProfilerException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.common.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.nd4j.common.primitives.Triple;

import static org.nd4j.linalg.indexing.NDArrayIndex.interval;

/**
 * Deep neural net normalization approach normalizes activations between layers
 * "brightness normalization"
 * Used for nets like AlexNet
 * <p>
 * For a^i_{x,y} the activity of a neuron computed by applying kernel i
 * at position (x,y) and applying ReLU nonlinearity, the response
 * normalized activation b^i_{x,y} is given by:
 * <p>
 * x^2 = (a^j_{x,y})^2
 * unitScale = (k + alpha * sum_{j=max(0, i - n/2)}^{max(N-1, i + n/2)} (a^j_{x,y})^2 )
 * y = b^i_{x,y} = x * unitScale**-beta
 * <p>
 * gy = epsilon (aka deltas from previous layer)
 * sumPart = sum(a^j_{x,y} * gb^j_{x,y})
 * gx = gy * unitScale**-beta - 2 * alpha * beta * sumPart/unitScale * a^i_{x,y}
 * <p>
 * Reference:<br>
 * <a href="http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf">http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf</a><br>
 * <a href="https://github.com/vlfeat/matconvnet/issues/10">https://github.com/vlfeat/matconvnet/issues/10</a><br>
 * <p>
 * Created by nyghtowl on 10/29/15.
 */
@Slf4j
public class LocalResponseNormalization
        extends AbstractLayer<org.deeplearning4j.nn.conf.layers.LocalResponseNormalization> {

    protected LocalResponseNormalizationHelper helper = null;
    protected int helperCountFail = 0;

    @Override
    public Layer clone() {
        return new LocalResponseNormalization(conf.clone(), dataType);
    }

    public LocalResponseNormalization(NeuralNetConfiguration conf, DataType dataType) {
        super(conf, dataType);
        initializeHelper();
    }

    void initializeHelper() {
        String backend = Nd4j.getExecutioner().getEnvironmentInformation().getProperty("backend");
        if("CUDA".equalsIgnoreCase(backend)) {
            helper = DL4JClassLoading.createNewInstance(
                    "org.deeplearning4j.cuda.normalization.CudnnLocalResponseNormalizationHelper",
                    LocalResponseNormalizationHelper.class,
                    dataType);
            log.debug("CudnnLocalResponseNormalizationHelper successfully initialized");
        }
        //2019-03-09 AB - MKL-DNN helper disabled: https://github.com/deeplearning4j/deeplearning4j/issues/7272
//        else if("CPU".equalsIgnoreCase(backend)){
//            helper = new MKLDNNLocalResponseNormalizationHelper();
//            log.debug("Created MKLDNNLocalResponseNormalizationHelper");
//        }
        if (helper != null && !helper.checkSupported(layerConf().getK(), layerConf().getN(), layerConf().getAlpha(), layerConf().getBeta())) {
            log.debug("Removed helper {} as not supported (k={}, n={}, alpha={}, beta={})", helper.getClass(), layerConf().getK(), layerConf().getN(), layerConf().getAlpha(), layerConf().getBeta());
            helper = null;
        }
    }

    @Override
    public double calcRegularizationScore(boolean backpropParamsOnly){
        return 0;
    }

    @Override
    public Type type() {
        return Type.NORMALIZATION;
    }

    @Override
    public void fit(INDArray input, LayerWorkspaceMgr workspaceMgr) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);

        double k = layerConf().getK();
        double n = layerConf().getN();
        double alpha = layerConf().getAlpha();
        double beta = layerConf().getBeta();
        int halfN = (int) n / 2;

        if (helper != null && (helperCountFail == 0 || !layerConf().isCudnnAllowFallback())){
            Pair<Gradient, INDArray> ret = null;
            try {
                ret = helper.backpropGradient(input, epsilon, k, n, alpha, beta, workspaceMgr);
            } catch (ND4JOpProfilerException e){
                throw e;    //NaN panic etc for debugging
            } catch (Throwable t){
                if(t.getMessage() != null && t.getMessage().contains("Failed to allocate")){
                    //This is a memory exception - don't fallback to built-in implementation
                    throw t;
                }
                if(layerConf().isCudnnAllowFallback()){
                    helperCountFail++;
                    log.warn("CuDNN LocalResponseNormalization backprop execution failed - falling back on built-in implementation",t);
                } else {
                    throw new RuntimeException("Error during LocalResponseNormalization CuDNN helper backprop - isCudnnAllowFallback() is set to false", t);
                }
            }
            if (ret != null) {
                return ret;
            }
        }

        boolean nchw = layerConf().getDataFormat() == CNN2DFormat.NCHW;
        int chDim = nchw ? 1 : 3;
        int hDim = nchw ? 2 : 1;
        int wDim = nchw ? 3 : 2;

        Triple<INDArray,INDArray,INDArray> triple = activateHelper(true, workspaceMgr, true);
        INDArray activations = triple.getFirst();
        INDArray unitScale = triple.getSecond();
        INDArray scale = triple.getThird();

        val channel = input.size(chDim);
        INDArray tmp, addVal;
        Gradient retGradient = new DefaultGradient();
        INDArray reverse = activations.mul(epsilon);
        INDArray sumPart = reverse.dup();

        // sumPart = sum(a^j_{x,y} * gb^j_{x,y})
        for (int i = 1; i < halfN + 1; i++) {
            if(nchw) {
                tmp = sumPart.get(NDArrayIndex.all(), interval(i, channel), NDArrayIndex.all(), NDArrayIndex.all());
                addVal = reverse.get(NDArrayIndex.all(), interval(0, channel - i), NDArrayIndex.all(), NDArrayIndex.all());
                sumPart.put(new INDArrayIndex[]{NDArrayIndex.all(), interval(i, channel), NDArrayIndex.all(),
                        NDArrayIndex.all()}, tmp.addi(addVal));

                tmp = sumPart.get(NDArrayIndex.all(), interval(0, channel - i), NDArrayIndex.all(), NDArrayIndex.all());
                addVal = reverse.get(NDArrayIndex.all(), interval(i, channel), NDArrayIndex.all(), NDArrayIndex.all());
                sumPart.put(new INDArrayIndex[]{NDArrayIndex.all(), interval(0, channel - i), NDArrayIndex.all(),
                        NDArrayIndex.all()}, tmp.addi(addVal));
            } else {
                tmp = sumPart.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), interval(i, channel));
                addVal = reverse.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), interval(0, channel - i));
                sumPart.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), interval(i, channel)}, tmp.addi(addVal));

                tmp = sumPart.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), interval(0, channel - i));
                addVal = reverse.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), interval(i, channel));
                sumPart.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), interval(0, channel - i)}, tmp.addi(addVal));
            }
        }

        // gx = gy * unitScale**-beta - 2 * alpha * beta * sumPart/unitScale * a^i_{x,y}    - rearranged for more in-place ops
        INDArray nextEpsilon = workspaceMgr.createUninitialized(ArrayType.ACTIVATION_GRAD, epsilon.dataType(), epsilon.shape(), epsilon.ordering());
        Nd4j.getExecutioner().exec(new MulOp(epsilon, scale, nextEpsilon));
        nextEpsilon.subi(sumPart.muli(input).divi(unitScale).muli(2 * alpha * beta));
        return new Pair<>(retGradient, nextEpsilon);
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        return activateHelper(training, workspaceMgr, false).getFirst();
    }

    private Triple<INDArray,INDArray,INDArray> activateHelper(boolean training, LayerWorkspaceMgr workspaceMgr, boolean forBackprop){
        assertInputSet(false);
        double k = layerConf().getK();
        double n = layerConf().getN();
        double alpha = layerConf().getAlpha();
        double beta = layerConf().getBeta();
        int halfN = (int) n / 2;

        if (helper != null && (helperCountFail == 0 || !layerConf().isCudnnAllowFallback())){
            INDArray activations = null;
            try {
                activations = helper.activate(input, training, k, n, alpha, beta, workspaceMgr);
            } catch (ND4JOpProfilerException e){
                throw e;    //NaN panic etc for debugging
            } catch (Throwable t){
                if(t.getMessage() != null && t.getMessage().contains("Failed to allocate")){
                    //This is a memory exception - don't fallback to built-in implementation
                    throw t;
                }

                if(layerConf().isCudnnAllowFallback()){
                    helperCountFail++;
                    log.warn("CuDNN LocalResponseNormalization backprop execution failed - falling back on built-in implementation",t);
                } else {
                    throw new RuntimeException("Error during LocalRsponseNormalization CuDNN helper backprop - isCudnnAllowFallback() is set to false", t);
                }
            }
            if (activations != null) {
                return new Triple<>(activations, null, null);
            }
        }

        boolean nchw = layerConf().getDataFormat() == CNN2DFormat.NCHW;
        int chDim = nchw ? 1 : 3;

        val channel = input.size(chDim);
        INDArray tmp, addVal;
        // x^2 = (a^j_{x,y})^2
        INDArray activitySqr = input.mul(input);
        INDArray sumPart = activitySqr.dup();

        //sum_{j=max(0, i - n/2)}^{max(N-1, i + n/2)} (a^j_{x,y})^2 )
        for (int i = 1; i < halfN + 1; i++) {

            if(nchw) {
                tmp = sumPart.get(NDArrayIndex.all(), interval(i, channel), NDArrayIndex.all(), NDArrayIndex.all());
                addVal = activitySqr.get(NDArrayIndex.all(), interval(0, channel - i), NDArrayIndex.all(),
                        NDArrayIndex.all());
                sumPart.put(new INDArrayIndex[]{NDArrayIndex.all(), interval(i, channel), NDArrayIndex.all(),
                        NDArrayIndex.all()}, tmp.addi(addVal));

                tmp = sumPart.get(NDArrayIndex.all(), interval(0, channel - i), NDArrayIndex.all(), NDArrayIndex.all());
                addVal = activitySqr.get(NDArrayIndex.all(), interval(i, channel), NDArrayIndex.all(), NDArrayIndex.all());
                sumPart.put(new INDArrayIndex[]{NDArrayIndex.all(), interval(0, channel - i), NDArrayIndex.all(),
                        NDArrayIndex.all()}, tmp.addi(addVal));
            } else {
                tmp = sumPart.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), interval(i, channel));
                addVal = activitySqr.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), interval(0, channel - i));
                sumPart.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), interval(i, channel)}, tmp.addi(addVal));

                tmp = sumPart.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), interval(0, channel - i));
                addVal = activitySqr.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), interval(i, channel));
                sumPart.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), interval(0, channel - i)}, tmp.addi(addVal));
            }
        }

        INDArray unitScale = null;
        INDArray scale = null;
        INDArray activations = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, input.dataType(), input.shape(), input.ordering());
        if(forBackprop) {
            // unitScale = (k + alpha * sum_{j=max(0, i - n/2)}^{max(N-1, i + n/2)} (a^j_{x,y})^2 )
            unitScale = sumPart.mul(alpha).addi(k);
            // y = x * unitScale**-beta
            scale = Transforms.pow(unitScale, -beta, true);
            Nd4j.getExecutioner().exec(new MulOp(input, scale, activations));
        } else {
            // unitScale = (k + alpha * sum_{j=max(0, i - n/2)}^{max(N-1, i + n/2)} (a^j_{x,y})^2 )
            sumPart.muli(alpha, activations).addi(k);
            Transforms.pow(activations, -beta, false);
            activations.muli(input);
        }
        if(forBackprop){
            return new Triple<>(activations, unitScale, scale);
        } else {
            return new Triple<>(activations, null, null);
        }
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    public void clearNoiseWeightParams() {
        //No op
    }

    @Override
    public LayerHelper getHelper() {
        return helper;
    }

    @Override
    public INDArray params() {
        return null;
    }

    @Override
    public INDArray getParam(String param) {
        return params();
    }

    @Override
    public void setParams(INDArray params) {

    }
}
