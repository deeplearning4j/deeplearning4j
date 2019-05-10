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

package org.datavec.hadoop.conf;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;

/**
 * Notes
 * 
 * https://linuxjunkies.wordpress.com/2011/11/21/a-hdfsclient-for-hadoop-using-the-native-java-api-a-tutorial/
 * 
 * Design Ideas
 * 
 * 	-	Need a DataVec Conf entry:
 * 		-	hadoop.configuration.path
 * 			-	example: hadoop.configuration.path=/home/hadoop/hadoop/conf/
 * 
 * 
 * @author josh
 *
 */
public class ConfigurationUtil {

    public static Configuration generateConfig(String baseConfPath) {

        String baseConfPathTrimmed = baseConfPath.trim();

        if (false == "/".equals(baseConfPathTrimmed.endsWith("/"))) {

            baseConfPathTrimmed += "/";

        }

        Configuration conf = new Configuration();
        conf.addResource(new Path(baseConfPathTrimmed + "core-site.xml"));
        conf.addResource(new Path(baseConfPathTrimmed + "hdfs-site.xml"));
        conf.addResource(new Path(baseConfPathTrimmed + "mapred-site.xml"));

        return conf;

    }

}
