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

package org.datavec.spark.util;

import lombok.NonNull;
import org.apache.hadoop.conf.Configuration;

import java.io.Serializable;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * A serializable version of {@link Configuration}
 * @author Alex Black
 */
public class SerializableHadoopConfig implements Serializable {

    private Map<String,String> content;
    private transient Configuration configuration;

    public SerializableHadoopConfig(@NonNull Configuration configuration){
        this.configuration = configuration;
        content = new LinkedHashMap<>();
        Iterator<Map.Entry<String,String>> iter = configuration.iterator();
        while(iter.hasNext()){
            Map.Entry<String,String> next = iter.next();
            content.put(next.getKey(), next.getValue());
        }
    }

    public synchronized Configuration getConfiguration(){
        if(configuration == null){
            configuration = new Configuration();
            for(Map.Entry<String,String> e : content.entrySet()){
                configuration.set(e.getKey(), e.getValue());
            }
        }
        return configuration;
    }

}
