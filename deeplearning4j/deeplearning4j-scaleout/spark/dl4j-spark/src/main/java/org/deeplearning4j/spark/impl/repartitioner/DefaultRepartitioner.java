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

package org.deeplearning4j.spark.impl.repartitioner;

import lombok.extern.slf4j.Slf4j;
import org.apache.spark.api.java.JavaRDD;
import org.deeplearning4j.spark.api.Repartitioner;
import org.deeplearning4j.spark.impl.common.CountPartitionsFunction;
import scala.Tuple2;

import java.util.List;

/**
 * DefaultRepartitioner: Repartition data so that we exactly the minimum number of objects per partition, up to a
 * specified maximum number of partitions partitions
 *
 * @author Alex Black
 */
@Slf4j
public class DefaultRepartitioner implements Repartitioner {
    public static final int DEFAULT_MAX_PARTITIONS = 5000;

    private final int maxPartitions;

    /**
     * Create a DefaultRepartitioner with the default maximum number of partitions, {@link #DEFAULT_MAX_PARTITIONS}
     */
    public DefaultRepartitioner(){
        this(DEFAULT_MAX_PARTITIONS);
    }

    /**
     *
     * @param maxPartitions Maximum number of partitions
     */
    public DefaultRepartitioner(int maxPartitions){
        this.maxPartitions = maxPartitions;
    }


    @Override
    public <T> JavaRDD<T> repartition(JavaRDD<T> rdd, int minObjectsPerPartition, int numExecutors) {
        //Num executors intentionally not used

        //Count each partition...
        List<Tuple2<Integer, Integer>> partitionCounts =
                rdd.mapPartitionsWithIndex(new CountPartitionsFunction<T>(), true).collect();
        int totalObjects = 0;
        for(Tuple2<Integer,Integer> t2 : partitionCounts){
            totalObjects += t2._2();
        }

        //Now, we want 'minObjectsPerPartition' in each partition... up to a maximum number of partitions
        int numPartitions;
        if(totalObjects / minObjectsPerPartition > maxPartitions){
            //Need more than the minimum, to avoid exceeding the maximum
            numPartitions = maxPartitions;
        } else {
            numPartitions = (int)Math.ceil(totalObjects / (double)minObjectsPerPartition);
        }
        return EqualRepartitioner.repartition(rdd, numPartitions, partitionCounts);
    }

    @Override
    public String toString(){
        return "DefaultRepartitioner(maxPartitions=" + maxPartitions + ")";
    }
}
