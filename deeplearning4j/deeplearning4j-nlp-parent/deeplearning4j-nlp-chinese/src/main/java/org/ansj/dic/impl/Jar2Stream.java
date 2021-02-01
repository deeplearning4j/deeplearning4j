/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.ansj.dic.impl;

import org.ansj.dic.DicReader;
import org.ansj.dic.PathToStream;
import org.ansj.exception.LibraryException;
import org.deeplearning4j.common.config.DL4JClassLoading;

import java.io.InputStream;

/**
 * 从系统jar包中读取文件，你们不能用，只有我能用 jar://org.ansj.dic.DicReader|/crf.model
 * 
 * @author ansj
 *
 */
public class Jar2Stream extends PathToStream {

    @Override
    public InputStream toStream(String path) {
        if (path.contains("|")) {
            String[] tokens = path.split("\\|");
            String className = tokens[0].substring(6);
            String resourceName = tokens[1].trim();

            Class<Object> resourceClass = DL4JClassLoading.loadClassByName(className);
            if (resourceClass == null) {
                throw new LibraryException(String.format("Class '%s' was not found.", className));
            }

            return resourceClass.getResourceAsStream(resourceName);
        } else {
            return DicReader.getInputStream(path.substring(6));
        }
    }

}
