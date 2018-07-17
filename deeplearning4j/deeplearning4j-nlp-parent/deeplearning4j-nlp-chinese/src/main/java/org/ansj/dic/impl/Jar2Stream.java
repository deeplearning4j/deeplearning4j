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

package org.ansj.dic.impl;

import org.ansj.dic.DicReader;
import org.ansj.dic.PathToStream;
import org.ansj.exception.LibraryException;

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
            String[] split = path.split("\\|");
            try {
                return Class.forName(split[0].substring(6)).getResourceAsStream(split[1].trim());
            } catch (ClassNotFoundException e) {
                throw new LibraryException(e);
            }
        } else {
            return DicReader.getInputStream(path.substring(6));
        }
    }

}
