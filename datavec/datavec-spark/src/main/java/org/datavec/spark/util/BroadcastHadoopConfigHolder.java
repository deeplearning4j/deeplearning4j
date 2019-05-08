/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.datavec.spark.util;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;

/**
 * This class holds a {@code Broadcast<SerializableHadoopConfig>} for re-use across multiple places.
 * The idea is that we often need spark configuration available for reading from (for example) HDFS directly, but Hadoop's
 * Configuration class is not serializable (hence using {@link SerializableHadoopConfig}; we also don't want to have
 * multiple copies of this in memory on each worker (Hadoop Configuration is immutable).
 *
 * @author Alex Black
 */
public class BroadcastHadoopConfigHolder {

    private static Broadcast<SerializableHadoopConfig> config;
    private static long sparkContextStartTime = -1; //Used to determine if spark context has changed - usually only use multiple spark contexts in tests etc

    private BroadcastHadoopConfigHolder(){ }

    public static Broadcast<SerializableHadoopConfig> get(JavaSparkContext sc){
        if(config != null && (!config.isValid() || sc.startTime() != sparkContextStartTime) ){
            config = null;
        }
        if(config != null){
            return config;
        }
        synchronized (BroadcastHadoopConfigHolder.class){
            if(config == null){
                config = sc.broadcast(new SerializableHadoopConfig(sc.hadoopConfiguration()));
                sparkContextStartTime = sc.startTime();
            }
        }
        return config;
    }
}
