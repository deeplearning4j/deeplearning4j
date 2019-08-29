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

package org.deeplearning4j.nn.conf.dropout;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.MulOp;
import org.nd4j.linalg.api.ops.random.impl.DropOutInverted;
import org.nd4j.linalg.exception.ND4JOpProfilerException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Implements standard (inverted) dropout.<br>
 * <br>
 * Regarding dropout probability. This is the probability of <it>retaining</it> each input activation value for a layer.
 * Thus, each input activation x is independently set to:<br>
 * x <- 0, with probability 1-p<br>
 * x <- x/p with probability p<br>
 * Note that this "inverted" dropout scheme maintains the expected value of activations - i.e., E(x) is the same before
 * and after dropout.<br>
 * Dropout schedules (i.e., varying probability p as a function of iteration/epoch) are also supported.<br>
 * <br>
 * Other libraries (notably, Keras) use p == probability(<i>dropping</i> an activation)<br>
 * In DL4J, {@code new Dropout(x)} will keep an input activation with probability x, and set to 0 with probability 1-x.<br>
 * Thus, a dropout value of 1.0 is functionally equivalent to no dropout: i.e., 100% probability of retaining
 * each input activation.<br>
 * <p>
 * Note 1: As per all IDropout instances, dropout is applied at training time only - and is automatically not applied at
 * test time (for evaluation, etc)<br>
 * Note 2: Care should be taken when setting lower (probability of retaining) values for (too much information may be
 * lost with aggressive (very low) dropout values).<br>
 * Note 3: Frequently, dropout is not applied to (or, has higher retain probability for) input (first layer)
 * layers. Dropout is also often not applied to output layers.<br>
 * Note 4: Implementation detail (most users can ignore): DL4J uses inverted dropout, as described here:
 * <a href="http://cs231n.github.io/neural-networks-2/">http://cs231n.github.io/neural-networks-2/</a>
 * </p>
 * <br>
 * See: Srivastava et al. 2014: Dropout: A Simple Way to Prevent Neural Networks from Overfitting
 * <a href="http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf">http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf</a>
 *
 * @author Alex Black
 */
@Data
@JsonIgnoreProperties({"mask", "helper", "helperCountFail"})
@EqualsAndHashCode(exclude = {"mask", "helper", "helperCountFail"})
@Slf4j
public class Dropout implements IDropout {

    /**
     * When using CuDNN and an error is encountered, should fallback to the non-CuDNN implementatation be allowed?
     * If set to false, an exception in CuDNN will be propagated back to the user. If false, the built-in
     * (non-CuDNN) implementation for LSTM/GravesLSTM will be used
     *
     */
    @Getter
    @Setter
    protected boolean helperAllowFallback = true;

    private double p;
    private ISchedule pSchedule;
    private transient INDArray mask;
    private transient DropoutHelper helper;
    private boolean initializedHelper = false;

    private int helperCountFail = 0;

    /**
     * @param activationRetainProbability Probability of retaining an activation - see {@link Dropout} javadoc
     */
    public Dropout(double activationRetainProbability) {
        this(activationRetainProbability, null);
        if(activationRetainProbability < 0.0){
            throw new IllegalArgumentException("Activation retain probability must be > 0. Got: " + activationRetainProbability);
        }
        if(activationRetainProbability == 0.0){
            throw new IllegalArgumentException("Invalid probability value: Dropout with 0.0 probability of retaining "
                    + "activations is not supported");
        }
    }

    /**
     * @param activationRetainProbabilitySchedule Schedule for probability of retaining an activation - see {@link Dropout} javadoc
     */
    public Dropout(ISchedule activationRetainProbabilitySchedule){
        this(Double.NaN, activationRetainProbabilitySchedule);
    }

    /**
     * When using a helper (CuDNN or MKLDNN in some cases) and an error is encountered, should fallback to the non-helper implementation be allowed?
     * If set to false, an exception in the helper will be propagated back to the user. If false, the built-in
     * (non-helper) implementation for Dropout will be used
     *
     * @param allowFallback Whether fallback to non-helper implementation should be used
     */
    public Dropout helperAllowFallback(boolean allowFallback) {
        this.setHelperAllowFallback(allowFallback);
        return this;
    }

    protected Dropout(@JsonProperty("p") double activationRetainProbability, @JsonProperty("pSchedule") ISchedule activationRetainProbabilitySchedule) {
        this.p = activationRetainProbability;
        this.pSchedule = activationRetainProbabilitySchedule;
    }

    /**
     * Initialize the CuDNN dropout helper, if possible
     */
    protected void initializeHelper(DataType dataType){
        String backend = Nd4j.getExecutioner().getEnvironmentInformation().getProperty("backend");
        if("CUDA".equalsIgnoreCase(backend)) {
            try {
                helper = Class.forName("org.deeplearning4j.nn.layers.dropout.CudnnDropoutHelper")
                        .asSubclass(DropoutHelper.class).getConstructor(DataType.class).newInstance(dataType);
                log.debug("CudnnDropoutHelper successfully initialized");
                if (!helper.checkSupported()) {
                    helper = null;
                }
            } catch (Throwable t) {
                if (!(t instanceof ClassNotFoundException)) {
                    log.warn("Could not initialize CudnnDropoutHelper", t);
                }
                //Unlike other layers, don't warn here about CuDNN not found - if the user has any other layers that can
                // benefit from them cudnn, they will get a warning from those
            }
        }
        initializedHelper = true;
    }


    @Override
    public INDArray applyDropout(INDArray inputActivations, INDArray output, int iteration, int epoch, LayerWorkspaceMgr workspaceMgr) {
        Preconditions.checkState(output.dataType().isFPType(), "Output array must be a floating point type, got %s for array of shape %ndShape",
                output.dataType(), output);
        double currP;
        if(pSchedule != null){
            currP = pSchedule.valueAt(iteration, epoch);
        } else {
            currP = p;
        }

        if(!initializedHelper){
            initializeHelper(output.dataType());
        }

        if(helper != null && (helperCountFail == 0 || !isHelperAllowFallback())){
            boolean helperWorked = false;
            try {
                helper.applyDropout(inputActivations, output, p);
                helperWorked = true;
            }catch (ND4JOpProfilerException e){
                throw e;    //NaN panic etc for debugging
            } catch (Exception e){
                if(e.getMessage().contains("Failed to allocate")){
                    //This is a memory exception - don't fallback to built-in implementation
                    throw e;
                }

                if(isHelperAllowFallback()){
                    helperCountFail++;
                    log.warn("CuDNN execution failed - falling back on built-in implementation",e);
                } else {
                    throw new RuntimeException("Error during Dropout CuDNN helper forward pass - helperAllowFallback() is set to false", e);
                }
            }

            if(helperWorked)
                return output;
        }

        INDArray inputCast = inputActivations;
        if(inputCast != output && inputCast.dataType() != output.dataType()){
            inputCast = inputCast.castTo(output.dataType());
        }

        mask = workspaceMgr.createUninitialized(ArrayType.INPUT, output.dataType(), output.shape(), output.ordering()).assign(1.0);
        Nd4j.getExecutioner().exec(new DropOutInverted(mask, mask, currP));
        Nd4j.getExecutioner().exec(new MulOp(inputCast, mask, output));
        return output;
    }

    @Override
    public INDArray backprop(INDArray gradAtOutput, INDArray gradAtInput, int iteration, int epoch) {
        if(helper != null && (helperCountFail == 0 || !isHelperAllowFallback())){
            boolean helperWorked = false;
            try {
                helper.backprop(gradAtOutput, gradAtInput);
                helperWorked = true;
            }catch (ND4JOpProfilerException e){
                throw e;    //NaN panic etc for debugging
            } catch (Exception e){
                if(e.getMessage().contains("Failed to allocate")){
                    //This is a memory exception - don't fallback to built-in implementation
                    throw e;
                }

                if(isHelperAllowFallback()){
                    helperCountFail++;
                    log.warn("CuDNN execution failed - falling back on built-in implementation",e);
                } else {
                    throw new RuntimeException("Error during Dropout CuDNN helper backprop - helperAllowFallback() is set to false", e);
                }
            }

            if(helperWorked)
                return gradAtInput;
        }

        Preconditions.checkState(mask != null, "Cannot perform backprop: Dropout mask array is absent (already cleared?)");
        //dL/dx = dL/dz * dz/dx, with z=0 or x/p
        //Mask already contains either 0 or 1/p, so just muli
        INDArray m = mask;
        if(m.dataType() != gradAtInput.dataType()){
            m = m.castTo(gradAtInput.dataType());
        }
        Nd4j.getExecutioner().exec(new MulOp(gradAtOutput, m, gradAtInput));
        mask = null;
        return gradAtInput;
    }

    @Override
    public void clear() {
        mask = null;
    }

    @Override
    public Dropout clone() {
        return new Dropout(p, pSchedule == null ? null : pSchedule.clone());
    }
}
