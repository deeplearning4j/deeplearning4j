/* ******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 * Copyright (c) 2019 Konduit K.K.
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

package org.deeplearning4j.nearestneighbor.server;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.clustering.sptree.DataPoint;
import org.deeplearning4j.clustering.vptree.VPTree;
import org.deeplearning4j.clustering.vptree.VPTreeFillSearch;
import org.deeplearning4j.nearestneighbor.client.NearestNeighborsClient;
import org.deeplearning4j.nearestneighbor.model.NearestNeighborRequest;
import org.deeplearning4j.nearestneighbor.model.NearestNeighborsResult;
import org.deeplearning4j.nearestneighbor.model.NearestNeighborsResults;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.serde.binary.BinarySerde;

import java.io.File;
import java.io.IOException;
import java.net.ServerSocket;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 4/27/17.
 */
public class NearestNeighborTest extends BaseDL4JTest {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Test
    public void testNearestNeighbor() {
        double[][] data = new double[][] {{1, 2, 3, 4}, {1, 2, 3, 5}, {3, 4, 5, 6}};
        INDArray arr = Nd4j.create(data);

        VPTree vpTree = new VPTree(arr, false);
        NearestNeighborRequest request = new NearestNeighborRequest();
        request.setK(2);
        request.setInputIndex(0);
        NearestNeighbor nearestNeighbor = NearestNeighbor.builder().tree(vpTree).points(arr).record(request).build();
        List<NearestNeighborsResult> results = nearestNeighbor.search();
        assertEquals(1, results.get(0).getIndex());
        assertEquals(2, results.size());

        assertEquals(1.0, results.get(0).getDistance(), 1e-4);
        assertEquals(4.0, results.get(1).getDistance(), 1e-4);
    }

    @Test
    public void testNearestNeighborInverted() {
        double[][] data = new double[][] {{1, 2, 3, 4}, {1, 2, 3, 5}, {3, 4, 5, 6}};
        INDArray arr = Nd4j.create(data);

        VPTree vpTree = new VPTree(arr, true);
        NearestNeighborRequest request = new NearestNeighborRequest();
        request.setK(2);
        request.setInputIndex(0);
        NearestNeighbor nearestNeighbor = NearestNeighbor.builder().tree(vpTree).points(arr).record(request).build();
        List<NearestNeighborsResult> results = nearestNeighbor.search();
        assertEquals(2, results.get(0).getIndex());
        assertEquals(2, results.size());

        assertEquals(-4.0, results.get(0).getDistance(), 1e-4);
        assertEquals(-1.0, results.get(1).getDistance(), 1e-4);
    }

    @Test
    public void vpTreeTest() throws Exception {
        INDArray matrix = Nd4j.rand(new int[] {400,10});
        INDArray rowVector = matrix.getRow(70);
        INDArray resultArr = Nd4j.zeros(400,1);
        Executor executor = Executors.newSingleThreadExecutor();
        VPTree vpTree = new VPTree(matrix);
        System.out.println("Ran!");
    }



    public static int getAvailablePort() {
        try {
            ServerSocket socket = new ServerSocket(0);
            try {
                return socket.getLocalPort();
            } finally {
                socket.close();
            }
        } catch (IOException e) {
            throw new IllegalStateException("Cannot find available port: " + e.getMessage(), e);
        }
    }

    @Test
    public void testServer() throws Exception {
        int localPort = getAvailablePort();
        Nd4j.getRandom().setSeed(7);
        INDArray rand = Nd4j.randn(10, 5);
        File writeToTmp = testDir.newFile();
        writeToTmp.deleteOnExit();
        BinarySerde.writeArrayToDisk(rand, writeToTmp);
        NearestNeighborsServer.runMain("--ndarrayPath", writeToTmp.getAbsolutePath(), "--nearestNeighborsPort",
                String.valueOf(localPort));

        Thread.sleep(3000);

        NearestNeighborsClient client = new NearestNeighborsClient("http://localhost:" + localPort);
        NearestNeighborsResults result = client.knnNew(5, rand.getRow(0));
        assertEquals(5, result.getResults().size());
        NearestNeighborsServer.getInstance().stop();
    }



    @Test
    public void testFullSearch() throws Exception {
        int numRows = 1000;
        int numCols = 100;
        int numNeighbors = 42;
        INDArray points = Nd4j.rand(numRows, numCols);
        VPTree tree = new VPTree(points);
        INDArray query = Nd4j.rand(new int[] {1, numCols});
        VPTreeFillSearch fillSearch = new VPTreeFillSearch(tree, numNeighbors, query);
        fillSearch.search();
        List<DataPoint> results = fillSearch.getResults();
        List<Double> distances = fillSearch.getDistances();
        assertEquals(numNeighbors, distances.size());
        assertEquals(numNeighbors, results.size());
    }

    @Test
    public void testDistances() {

        INDArray indArray = Nd4j.create(new float[][]{{3, 4}, {1, 2}, {5, 6}});
        INDArray record = Nd4j.create(new float[][]{{7, 6}});
        VPTree vpTree = new VPTree(indArray, "euclidean", false);
        VPTreeFillSearch vpTreeFillSearch = new VPTreeFillSearch(vpTree, 3, record);
        vpTreeFillSearch.search();
        //System.out.println(vpTreeFillSearch.getResults());
        System.out.println(vpTreeFillSearch.getDistances());
    }
}
