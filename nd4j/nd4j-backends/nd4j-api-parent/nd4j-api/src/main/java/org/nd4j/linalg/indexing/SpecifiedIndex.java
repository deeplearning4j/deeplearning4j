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

package org.nd4j.linalg.indexing;

import org.nd4j.shade.guava.primitives.Longs;
import lombok.Data;
import net.ericaro.neoitertools.Generator;
import net.ericaro.neoitertools.Itertools;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.LongUtils;

import java.util.List;
import java.util.NoSuchElementException;

/**
 * @author Adam Gibson
 */
@Data
public class SpecifiedIndex implements INDArrayIndex {
    private long[] indexes;

    public SpecifiedIndex(int... indexes) {
        this.indexes = LongUtils.toLongs(indexes);
    }

    public SpecifiedIndex(long... indexes) {
        this.indexes = indexes;
    }

    @Override
    public long end() {
        return indexes[indexes.length - 1];
    }

    @Override
    public long offset() {
        return indexes[0];
    }

    @Override
    public long length() {
        return indexes.length;
    }

    @Override
    public long stride() {
        return 1;
    }

    @Override
    public void reverse() {

    }

    @Override
    public boolean isInterval() {
        return false;
    }

    @Override
    public void init(INDArray arr, long begin, int dimension) {

    }

    @Override
    public void init(INDArray arr, int dimension) {

    }

    @Override
    public void init(long begin, long end, long max) {

    }

    @Override
    public void init(long begin, long end) {

    }


    /**
     * Iterate over a cross product of the
     * coordinates
     * @param indexes the coordinates to iterate over.
     *                Each element of the array should be of opType {@link SpecifiedIndex}
     *                otherwise it will end up throwing an exception
     * @return the generator for iterating over all the combinations of the specified indexes.
     */
    public static Generator<List<List<Long>>> iterate(INDArrayIndex... indexes) {
        Generator<List<List<Long>>> gen = Itertools.product(new SpecifiedIndexesGenerator(indexes));
        return gen;
    }

    /**
     * Iterate over a cross product of the
     * coordinates
     * @param indexes the coordinates to iterate over.
     *                Each element of the array should be of opType {@link SpecifiedIndex}
     *                otherwise it will end up throwing an exception
     * @return the generator for iterating over all the combinations of the specified indexes.
     */
    public static Generator<List<List<Long>>> iterateOverSparse(INDArrayIndex... indexes) {
        Generator<List<List<Long>>> gen = Itertools.product(new SparseSpecifiedIndexesGenerator(indexes));
        return gen;
    }


    /**
     * A generator for {@link SpecifiedIndex} for
     * {@link Itertools#product(Generator)}
     *    to iterate
     over an array given a set of  iterators
     */
    public static class SpecifiedIndexesGenerator implements Generator<Generator<List<Long>>> {
        private int index = 0;
        private INDArrayIndex[] indexes;

        /**
         * The indexes to generate from
         * @param indexes the indexes to generate from
         */
        public SpecifiedIndexesGenerator(INDArrayIndex[] indexes) {
            this.indexes = indexes;
            for(int i=0; i<indexes.length; i++ ){
                //Replace point indices with specified indices
                if(indexes[i] instanceof PointIndex){
                    indexes[i] = new SpecifiedIndex(indexes[i].offset());
                }
            }
        }

        @Override
        public Generator<List<Long>> next() throws NoSuchElementException {
            if (index >= indexes.length) {
                throw new NoSuchElementException("Done");
            }

            SpecifiedIndex specifiedIndex = (SpecifiedIndex) indexes[index++];
            Generator<List<Long>> ret = specifiedIndex.generator();
            return ret;
        }
    }

    /**
     * A generator for {@link SpecifiedIndex} for
     * {@link Itertools#product(Generator)}
     *    to iterate
     over an array given a set of  iterators
     */
    public static class SparseSpecifiedIndexesGenerator implements Generator<Generator<List<Long>>> {
        private int index = 0;
        private INDArrayIndex[] indexes;

        /**
         * The indexes to generate from
         * @param indexes the indexes to generate from
         */
        public SparseSpecifiedIndexesGenerator(INDArrayIndex[] indexes) {
            this.indexes = indexes;
        }

        @Override
        public Generator<List<Long>> next() throws NoSuchElementException {
            if (index >= indexes.length) {
                throw new NoSuchElementException("Done");
            }

            SpecifiedIndex specifiedIndex = (SpecifiedIndex) indexes[index++];
            Generator<List<Long>> ret = specifiedIndex.sparseGenerator();
            return ret;
        }
    }


    public class SingleGenerator implements Generator<List<Long>> {
        /**
         * @return the next item in the sequence.
         * @throws NoSuchElementException when sequence is exhausted.
         */
        @Override
        public List<Long> next() throws NoSuchElementException {
//            if (!SpecifiedIndex.this.hasNext())
//                throw new NoSuchElementException();
//
//            return Longs.asList(SpecifiedIndex.this.next());
            throw new RuntimeException();
        }
    }
    public class SparseSingleGenerator implements Generator<List<Long>> {
        /**
         * @return the next item in the sequence.
         * @throws NoSuchElementException when sequence is exhausted.
         */
        @Override
        public List<Long> next() throws NoSuchElementException {
//            if (!SpecifiedIndex.this.hasNext())
//                throw new NoSuchElementException();
//            long[] pair = SpecifiedIndex.this.nextSparse();
//            return Arrays.asList(pair[0], pair[1]);
            throw new RuntimeException();
        }
    }

    public Generator<List<Long>> generator() {
        return new SingleGenerator();
    }

    public Generator<List<Long>> sparseGenerator() {
        return new SparseSingleGenerator();
    }

}
