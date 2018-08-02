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

package org.deeplearning4j.clustering.lsh;

import lombok.Getter;
import lombok.val;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastDivOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastEqualTo;
import org.nd4j.linalg.api.ops.impl.transforms.Sign;

import org.nd4j.linalg.api.ops.random.impl.GaussianDistribution;
import org.nd4j.linalg.api.ops.random.impl.UniformDistribution;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;


/**
 * This class implements Entropy LSH for the cosine distance, in order to preserve memory for large datasets.
 *
 * Entropy SLH is the LSH scheme of
 *
 * _Entropy based nearest neighbor search in high dimensions_
 * R Panigrahy - SIAM 2006
 * https://arxiv.org/pdf/cs/0510019.pdf
 *
 * To read more about LSH, in particular for the Cosine distance, see
 * chapter 3 of :
 * _Mining Massive Datasets_, Anand Rajaraman and Jeffrey Ullman
 * http://www.mmds.org/
 *
 * The original development of LSH for the cosine distance is from
 * Similarity estimation techniques from rounding algorithms
 * MS Charikar - STOCS, 2002
 *
 * Note for high-precision or distributed settings, you should not
 * use this and rather extend this to layered LSH ( https://arxiv.org/abs/1210.7057 )
 *
 */
public class RandomProjectionLSH implements LSH {

    @Override
    public String getDistanceMeasure(){
        return "cosinedistance";
    }

    @Getter private int hashLength;

    @Getter private int numTables;

    @Getter private int inDimension;


    @Getter private double radius;

    INDArray randomProjection;

    INDArray index;

    INDArray indexData;


    private INDArray gaussianRandomMatrix(int[] shape, Random rng){
        INDArray res = Nd4j.create(shape);

        GaussianDistribution op1 = new GaussianDistribution(res, 0.0, 1.0 / Math.sqrt(shape[0]));

        Nd4j.getExecutioner().exec(op1, rng);
        return res;
    }

    public RandomProjectionLSH(int hashLength, int numTables, int inDimension, double radius){
        this(hashLength, numTables, inDimension, radius, Nd4j.getRandom());
    }

    /**
     * Creates a locality-sensitive hashing index for the cosine distance,
     * a (d1, d2, (180 − d1)/180,(180 − d2)/180)-sensitive hash family before amplification
     *
     * @param hashLength the length of the compared hash in an AND construction,
     * @param numTables the entropy-equivalent of a nb of hash tables in an OR construction, implemented here with the multiple
     *                  probes of Panigraphi (op. cit).
     * @param inDimension the dimendionality of the points being indexed
     * @param radius the radius of points to generate probes for. Instead of using multiple physical hash tables in an OR construction
     * @param rng a Random object to draw samples from
     */
    public RandomProjectionLSH(int hashLength, int numTables, int inDimension, double radius, Random rng){
        this.hashLength = hashLength;
        this.numTables = numTables;
        this.inDimension = inDimension;
        this.radius = radius;
        randomProjection = gaussianRandomMatrix(new int[]{inDimension, hashLength}, rng);
    }

    /**
     * This picks uniformaly distributed random points on the unit of a sphere using the method of:
     *
     * An efficient method for generating uniformly distributed points on the surface of an n-dimensional sphere
     * JS Hicks, RF Wheeling - Communications of the ACM, 1959
     * @param data a query to generate multiple probes for
     * @return `numTables`
     */
    public INDArray entropy(INDArray data){

        INDArray data2 =
                    Nd4j.getExecutioner().exec(new GaussianDistribution(Nd4j.create(numTables, inDimension), radius));

        INDArray norms = Nd4j.norm2(data2.dup(), -1);

        assert(norms.shape()[0] == numTables);
        assert(norms.shape()[1] == 1);

        data2.diviColumnVector(norms);
        data2.addiRowVector(data);
        return data2;
    }

    /**
     * Returns hash values for a particular query
     * @param data a query vector
     * @return its hashed value
     */
    public INDArray hash(INDArray data) {
        if (data.shape()[1] != inDimension){
            throw new ND4JIllegalStateException(
                    String.format("Invalid shape: Requested INDArray shape %s, this table expects dimension %d",
                            Arrays.toString(data.shape()), inDimension));
        }
        INDArray projected = data.mmul(randomProjection);
        INDArray res = Nd4j.getExecutioner().execAndReturn(new Sign(projected));
        return res;
    }

    /**
     * Populates the index. Beware, not incremental, any further call replaces the index instead of adding to it.
     * @param data the vectors to index
     */
    @Override
    public void makeIndex(INDArray data) {
        index = hash(data);
        indexData = data;
    }

    // data elements in the same bucket as the query, without entropy
    INDArray rawBucketOf(INDArray query){
        INDArray pattern = hash(query);

        INDArray res = Nd4j.zeros(index.shape());
        Nd4j.getExecutioner().exec(new BroadcastEqualTo(index, pattern, res, -1));
        return res.min(-1);
    }

    @Override
    public INDArray bucket(INDArray query) {
        INDArray queryRes = rawBucketOf(query);

        if(numTables > 1) {
            INDArray entropyQueries = entropy(query);

            // loop, addi + conditionalreplace -> poor man's OR function
            for (int i = 0; i < numTables; i++) {
                INDArray row = entropyQueries.getRow(i);
                queryRes.addi(rawBucketOf(row));
            }
            BooleanIndexing.replaceWhere(queryRes, 1.0, Conditions.greaterThan(0.0));
        }

        return queryRes;
    }

    // data elements in the same entropy bucket as the query,
    INDArray bucketData(INDArray query){
        INDArray mask = bucket(query);
        int nRes = mask.sum(0).getInt(0);
        INDArray res = Nd4j.create(new int[] {nRes, inDimension});
        int j = 0;
        for (int i = 0; i < nRes; i++){
            while (mask.getInt(j) == 0 && j < mask.length() - 1) {
                j += 1;
            }
            if (mask.getInt(j) == 1) res.putRow(i, indexData.getRow(j));
            j += 1;
        }
        return res;
    }

    @Override
    public INDArray search(INDArray query, double maxRange) {
        if (maxRange < 0)
            throw new IllegalArgumentException("ANN search should have a positive maximum search radius");

        INDArray bucketData = bucketData(query);
        INDArray distances = Transforms.allCosineDistances(bucketData, query, -1);
        INDArray[] idxs = Nd4j.sortWithIndices(distances, -1, true);

        INDArray shuffleIndexes = idxs[0];
        INDArray sortedDistances = idxs[1];
        int accepted = 0;
        while (accepted < sortedDistances.length() && sortedDistances.getInt(accepted) <= maxRange) accepted +=1;

        INDArray res = Nd4j.create(new int[] {accepted, inDimension});
        for(int i = 0; i < accepted; i++){
            res.putRow(i, bucketData.getRow(shuffleIndexes.getInt(i)));
        }
        return res;
    }

    @Override
    public INDArray search(INDArray query, int k) {
        if (k < 1)
            throw new IllegalArgumentException("An ANN search for k neighbors should at least seek one neighbor");

        INDArray bucketData = bucketData(query);
        INDArray distances = Transforms.allCosineDistances(bucketData, query, -1);
        INDArray[] idxs = Nd4j.sortWithIndices(distances, -1, true);

        INDArray shuffleIndexes = idxs[0];
        INDArray sortedDistances = idxs[1];
        val accepted = Math.min(k, sortedDistances.shape()[1]);

        INDArray res = Nd4j.create(accepted, inDimension);
        for(int i = 0; i < accepted; i++){
            res.putRow(i, bucketData.getRow(shuffleIndexes.getInt(i)));
        }
        return res;
    }
}
