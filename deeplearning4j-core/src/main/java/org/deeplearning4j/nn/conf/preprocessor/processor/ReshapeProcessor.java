/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.nn.conf.preprocessor.processor;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.conf.preprocessor.output.BaseOutputPostProcessor;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;

/**Reshape post processor.<br>
 * Used to reshape activations on forward pass.<br>
 * Also (optionally, if fromShape != null) used to reshape, weights*deltas
 * during backward pass. Otherwise, no changes are made during backward pass
 * 
 * @author Adam Gibson
 */
public class ReshapeProcessor extends BaseProcessor {
	private int[] toShape;		//Activations: To this shape in forward pass
	private int[] fromShape;	//Epsilons: To this shape in backward pass
    

	/**@param toShape The shape that activations are reshaped to
	 * @param fromShape May be null. If null: no change/op during backward pass.
	 * Otherwise fromShape is the shape that epsilons (weights*deltas or equiv.)
	 *  are reshaped to by backprop(...)
	 */
    public ReshapeProcessor(int[] toShape, int[] fromShape){
    	this.toShape = toShape;
    	this.fromShape = fromShape;
    }

    public ReshapeProcessor(int... toShape) {
        this(toShape,null);
    }

    @Override
    public INDArray process(INDArray output) {
        return output.reshape(toShape);
    }

    @Override
    public Pair<Gradient,INDArray> backprop(Pair<Gradient,INDArray> input) {
    	if( fromShape == null ) return input;	//no-op
    	return new Pair<>(input.getFirst(), input.getSecond().reshape(fromShape));
    }
}
