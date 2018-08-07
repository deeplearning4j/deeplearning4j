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

package org.deeplearning4j.datasets.iterator;

import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Triple;

import java.util.*;

/**
 * RandomMultiDataSetIterator: Generates random values (or zeros, ones, integers, etc) according to some distribution.<br>
 * Note: This is typically used for testing, debugging and benchmarking purposes.
 *
 * @author Alex Black
 */
public class RandomMultiDataSetIterator implements MultiDataSetIterator {

    public enum Values {RANDOM_UNIFORM, RANDOM_NORMAL, ONE_HOT, ZEROS, ONES, BINARY, INTEGER_0_10, INTEGER_0_100, INTEGER_0_1000,
        INTEGER_0_10000, INTEGER_0_100000}

    private final int numMiniBatches;
    private final List<Triple<long[], Character, Values>> features;
    private final List<Triple<long[], Character, Values>> labels;
    @Getter @Setter
    private MultiDataSetPreProcessor preProcessor;

    private int position;

    public RandomMultiDataSetIterator(int numMiniBatches, @NonNull List<Triple<long[], Character, Values>> features, @NonNull List<Triple<long[], Character, Values>> labels){
        Preconditions.checkArgument(numMiniBatches > 0, "Number of minibatches must be positive: got %s", numMiniBatches);
        Preconditions.checkArgument(features.size() > 0, "No features defined");
        Preconditions.checkArgument(labels.size() > 0, "No labels defined");

        this.numMiniBatches = numMiniBatches;
        this.features = features;
        this.labels = labels;
    }

    @Override
    public MultiDataSet next(int i) {
        return next();
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        position = 0;
    }

    @Override
    public boolean hasNext() {
        return position < numMiniBatches;
    }

    @Override
    public MultiDataSet next() {
        if(!hasNext())
            throw new NoSuchElementException("No next element");
        INDArray[] f = new INDArray[features.size()];
        INDArray[] l = new INDArray[labels.size()];

        for( int i=0; i<f.length; i++ ){
            Triple<long[], Character, Values> t = features.get(i);
            f[i] = generate(t.getFirst(), t.getSecond(), t.getThird());
        }

        for( int i=0; i<l.length; i++ ){
            Triple<long[], Character, Values> t = labels.get(i);
            l[i] = generate(t.getFirst(), t.getSecond(), t.getThird());
        }

        position++;
        MultiDataSet mds = new org.nd4j.linalg.dataset.MultiDataSet(f,l);
        if(preProcessor != null)
            preProcessor.preProcess(mds);
        return mds;
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException("Not supported");
    }

    public static class Builder {

        private int numMiniBatches;
        private List<Triple<long[], Character, Values>> features = new ArrayList<>();
        private List<Triple<long[], Character, Values>> labels = new ArrayList<>();

        /**
         * @param numMiniBatches Number of minibatches per epoch
         */
        public Builder(int numMiniBatches){
            this.numMiniBatches = numMiniBatches;
        }

        /**
         * Add a new features array to the iterator
         * @param shape  Shape of the features
         * @param values Values to fill the array with
         */
        public Builder addFeatures(long[] shape, Values values) {
            return addFeatures(shape, 'c', values);
        }

        /**
         * Add a new features array to the iterator
         * @param shape  Shape of the features
         * @param order  Order ('c' or 'f') for the array
         * @param values Values to fill the array with
         */
        public Builder addFeatures(long[] shape, char order, Values values){
            features.add(new Triple<>(shape, order, values));
            return this;
        }

        /**
         * Add a new labels array to the iterator
         * @param shape  Shape of the features
         * @param values Values to fill the array with
         */
        public Builder addLabels(long[] shape, Values values) {
            return addLabels(shape, 'c', values);
        }

        /**
         * Add a new labels array to the iterator
         * @param shape  Shape of the features
         * @param order  Order ('c' or 'f') for the array
         * @param values Values to fill the array with
         */
        public Builder addLabels(long[] shape, char order, Values values){
            labels.add(new Triple<>(shape, order, values));
            return this;
        }

        public RandomMultiDataSetIterator build(){
            return new RandomMultiDataSetIterator(numMiniBatches, features, labels);
        }
    }

    /**
     * Generate a random array with the specified shape
     * @param shape  Shape of the array
     * @param values Values to fill the array with
     * @return Random array of specified shape + contents
     */
    public static INDArray generate(long[] shape, Values values) {
        return generate(shape, Nd4j.order(), values);
    }

    /**
     * Generate a random array with the specified shape and order
     * @param shape  Shape of the array
     * @param order  Order of array ('c' or 'f')
     * @param values Values to fill the array with
     * @return Random array of specified shape + contents
     */
    public static INDArray generate(long[] shape, char order, Values values){
        switch (values){
            case RANDOM_UNIFORM:
                return Nd4j.rand(Nd4j.createUninitialized(shape,order));
            case RANDOM_NORMAL:
                return Nd4j.randn(Nd4j.createUninitialized(shape,order));
            case ONE_HOT:
                Random r = new Random(Nd4j.getRandom().nextLong());
                INDArray out = Nd4j.create(shape,order);
                if(shape.length == 1){
                    out.putScalar(r.nextInt((int) shape[0]), 1.0);
                } else if(shape.length == 2){
                    for( int i=0; i<shape[0]; i++ ){
                        out.putScalar(i, r.nextInt((int) shape[1]), 1.0);
                    }
                } else if(shape.length == 3){
                    for( int i=0; i<shape[0]; i++ ){
                        for(int j=0; j<shape[2]; j++ ){
                            out.putScalar(i, r.nextInt((int) shape[1]), j, 1.0);
                        }
                    }
                } else if(shape.length == 4){
                    for( int i=0; i<shape[0]; i++ ){
                        for(int j=0; j<shape[2]; j++ ){
                            for(int k=0; k<shape[3]; k++ ) {
                                out.putScalar(i, r.nextInt((int) shape[1]), j, k, 1.0);
                            }
                        }
                    }
                } else if(shape.length == 5){
                    for( int i=0; i<shape[0]; i++ ){
                        for(int j=0; j<shape[2]; j++ ){
                            for(int k=0; k<shape[3]; k++ ) {
                                for( int l=0; l<shape[4]; l++ ) {
                                    out.putScalar(new int[]{i, r.nextInt((int) shape[1]), j, k, l}, 1.0);
                                }
                            }
                        }
                    }
                } else {
                    throw new RuntimeException("Not supported: rank 6+ arrays. Shape: " + Arrays.toString(shape));
                }
                return out;
            case ZEROS:
                return Nd4j.create(shape,order);
            case ONES:
                return Nd4j.createUninitialized(shape,order).assign(1.0);
            case BINARY:
                return Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.createUninitialized(shape, order), 0.5));
            case INTEGER_0_10:
                return Transforms.floor(Nd4j.rand(shape).muli(10), false);
            case INTEGER_0_100:
                return Transforms.floor(Nd4j.rand(shape).muli(100), false);
            case INTEGER_0_1000:
                return Transforms.floor(Nd4j.rand(shape).muli(1000), false);
            case INTEGER_0_10000:
                return Transforms.floor(Nd4j.rand(shape).muli(10000), false);
            case INTEGER_0_100000:
                return Transforms.floor(Nd4j.rand(shape).muli(100000), false);
            default:
                throw new RuntimeException("Unknown enum value: " + values);

        }
    }

}
