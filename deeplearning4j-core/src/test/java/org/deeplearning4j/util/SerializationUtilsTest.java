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

package org.deeplearning4j.util;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.clustering.cluster.Cluster;
import org.deeplearning4j.clustering.cluster.ClusterSet;
import org.deeplearning4j.clustering.cluster.Point;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;

import static org.junit.Assert.assertEquals;

/**
 * Created by mjk on 9/15/14.
 */
public class SerializationUtilsTest {
    @Test
    public void testWriteRead() {
        DataSetIterator iter = new IrisDataSetIterator(150, 150);
        String irisData = "irisData.dat";

        DataSet freshDataSet = iter.next(150);

        SerializationUtils.saveObject(freshDataSet, new File(irisData));

        DataSet readDataSet = SerializationUtils.readObject(new File(irisData));

        assertEquals(freshDataSet.getFeatureMatrix(),readDataSet.getFeatureMatrix());
        assertEquals(freshDataSet.getLabels(), readDataSet.getLabels());
        try {
            FileUtils.forceDelete(new File(irisData));
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    @Test
    public void testClusterSetWriteRead() {
        ClusterSet set = new ClusterSet();
        String filename = "clusterSet.dat";

        String clusterId = "test";
        Point clusterCenter = new Point(clusterId, Nd4j.linspace(0, 2, 3));

        Cluster cluster = set.addNewClusterWithCenter(clusterCenter);

        SerializationUtils.saveObject(set, new File(filename));

        ClusterSet freshSet = SerializationUtils.readObject(new File(filename));

        Cluster freshCluster = freshSet.getCluster(cluster.getId());

        assertEquals(freshCluster.getLabel(), cluster.getLabel());
        assertEquals(freshCluster.getCenter().getArray(), cluster.getCenter().getArray());

        try {
            FileUtils.forceDelete(new File(filename));
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
