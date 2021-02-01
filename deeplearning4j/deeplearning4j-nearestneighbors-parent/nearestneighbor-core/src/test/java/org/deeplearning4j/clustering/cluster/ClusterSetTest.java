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

package org.deeplearning4j.clustering.cluster;

import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

public class ClusterSetTest {
    @Test
    public void testGetMostPopulatedClusters() {
        ClusterSet clusterSet = new ClusterSet(false);
        List<Cluster> clusters = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            Cluster cluster = new Cluster();
            cluster.setPoints(Point.toPoints(Nd4j.randn(i + 1, 5)));
            clusters.add(cluster);
        }
        clusterSet.setClusters(clusters);
        List<Cluster> mostPopulatedClusters = clusterSet.getMostPopulatedClusters(5);
        for (int i = 0; i < 5; i++) {
            Assert.assertEquals(5 - i, mostPopulatedClusters.get(i).getPoints().size());
        }
    }
}
