package org.nd4j.linalg.indexing;

import com.google.common.primitives.Ints;
import lombok.Data;
import net.ericaro.neoitertools.Generator;
import net.ericaro.neoitertools.Itertools;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;
import java.util.NoSuchElementException;

/**
 * @author Adam Gibson
 */
@Data
public class SpecifiedIndex implements INDArrayIndex {
    private int[] indexes;
    private int counter = 0;

    public SpecifiedIndex(int... indexes) {
        this.indexes = indexes;
    }

    @Override
    public int end() {
        return indexes[indexes.length - 1];
    }

    @Override
    public int offset() {
        return indexes[0];
    }

    @Override
    public int length() {
        return indexes.length;
    }

    @Override
    public int stride() {
        return 1;
    }

    @Override
    public int current() {
        return indexes[counter - 1];
    }

    @Override
    public boolean hasNext() {
        return counter < indexes.length;
    }

    @Override
    public int next() {
        return indexes[counter++];
    }

    @Override
    public void reverse() {

    }

    @Override
    public boolean isInterval() {
        return false;
    }

    @Override
    public void setInterval(boolean isInterval) {

    }

    @Override
    public void init(INDArray arr, int begin, int dimension) {

    }

    @Override
    public void init(INDArray arr, int dimension) {

    }

    @Override
    public void init(int begin, int end) {

    }

    @Override
    public void reset() {
        counter = 0;
    }


    /**
     * Iterate over a cross product of the
     * coordinates
     * @param indexes the coordinates to iterate over.
     *                Each element of the array should be of type {@link SpecifiedIndex}
     *                otherwise it will end up throwing an exception
     * @return the generator for iterating over all the combinations of the specified indexes.
     */
    public static Generator<List<List<Integer>>> iterate(INDArrayIndex... indexes) {
        Generator<List<List<Integer>>> gen = Itertools.product(new SpecifiedIndexesGenerator(indexes));
        return gen;
    }


    /**
     * A generator for {@link SpecifiedIndex} for
     * {@link Itertools#product(Generator)}
     *    to iterate
     over an array given a set of  iterators
     */
    public static class SpecifiedIndexesGenerator implements Generator<Generator<List<Integer>>> {
        private int index = 0;
        private INDArrayIndex[] indexes;

        /**
         * The indexes to generate from
         * @param indexes the indexes to generate from
         */
        public SpecifiedIndexesGenerator(INDArrayIndex[] indexes) {
            this.indexes = indexes;
        }

        @Override
        public Generator<List<Integer>> next() throws NoSuchElementException {
            if (index >= indexes.length) {
                throw new NoSuchElementException("Done");
            }

            SpecifiedIndex specifiedIndex = (SpecifiedIndex) indexes[index++];
            Generator<List<Integer>> ret = specifiedIndex.generator();
            return ret;
        }
    }



    public class SingleGenerator implements Generator<List<Integer>> {
        /**
         * @return the next item in the sequence.
         * @throws NoSuchElementException when sequence is exhausted.
         */
        @Override
        public List<Integer> next() throws NoSuchElementException {
            if (!SpecifiedIndex.this.hasNext())
                throw new NoSuchElementException();

            return Ints.asList(SpecifiedIndex.this.next());
        }
    }

    public Generator<List<Integer>> generator() {
        return new SingleGenerator();
    }

}
