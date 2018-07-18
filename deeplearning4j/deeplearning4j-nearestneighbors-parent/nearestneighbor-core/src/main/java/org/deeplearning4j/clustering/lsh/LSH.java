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

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseAccumulation;

/**
 * This interface gathers the minimal elements for an LSH implementation
 *
 * See chapter 3 of :
 * _Mining Massive Datasets_, Anand Rajaraman and Jeffrey Ullman
 * http://www.mmds.org/
 *
 */
public interface LSH {

    /**
     * Returns an instance of the distance measure associated to the LSH family of this implementation.
     * Beware, hashing families and their amplification constructs are distance-specific.
     */
     String getDistanceMeasure();

    /**
     * Returns the size of a hash compared against in one hashing bucket, corresponding to an AND construction
     *
     * denoting hashLength by h,
     * amplifies a (d1, d2, p1, p2) hash family into a
     *                   (d1, d2, p1^h, p2^h)-sensitive one (match probability is decreasing with h)
     *
     * @return the length of the hash in the AND construction used by this index
     */
     int getHashLength();

    /**
     *
     * denoting numTables by n,
     * amplifies a (d1, d2, p1, p2) hash family into a
     *                   (d1, d2, (1-p1^n), (1-p2^n))-sensitive one (match probability is increasing with n)
     *
     * @return the # of hash tables in the OR construction used by this index
     */
     int getNumTables();

    /**
     * @return The dimension of the index vectors and queries
     */
     int getInDimension();

    /**
     * Populates the index with data vectors.
     * @param data the vectors to index
     */
     void makeIndex(INDArray data);

    /**
     * Returns the set of all vectors that could approximately be considered negihbors of the query,
     * without selection on the basis of distance or number of neighbors.
     * @param query a  vector to find neighbors for
     * @return its approximate neighbors, unfiltered
     */
     INDArray bucket(INDArray query);

    /**
     * Returns the approximate neighbors within a distance bound.
     * @param query a vector to find neighbors for
     * @param maxRange the maximum distance between results and the query
     * @return approximate neighbors within the distance bounds
     */
     INDArray search(INDArray query, double maxRange);

    /**
     * Returns the approximate neighbors within a k-closest bound
     * @param query a vector to find neighbors for
     * @param k the maximum number of closest neighbors to return
     * @return at most k neighbors of the query, ordered by increasing distance
     */
     INDArray search(INDArray query, int k);
}
