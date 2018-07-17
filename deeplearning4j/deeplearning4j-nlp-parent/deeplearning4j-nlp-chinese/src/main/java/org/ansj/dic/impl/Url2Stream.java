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

import org.ansj.dic.PathToStream;
import org.ansj.exception.LibraryException;

import java.io.InputStream;
import java.net.URL;

/**
 * url://http://maven.nlpcn.org/down/library/default.dic
 * 
 * @author ansj
 *
 */
public class Url2Stream extends PathToStream {

    @Override
    public InputStream toStream(String path) {
        try {
            URL url = new URL(path);
            return url.openStream();
        } catch (Exception e) {
            throw new LibraryException("err to load by http " + path + " message : " + e.getMessage());
        }

    }

}
