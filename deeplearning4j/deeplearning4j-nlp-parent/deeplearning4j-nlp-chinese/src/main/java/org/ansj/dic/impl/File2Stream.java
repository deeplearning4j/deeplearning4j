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
import org.nlpcn.commons.lang.util.logging.Log;
import org.nlpcn.commons.lang.util.logging.LogFactory;

import java.io.*;
import java.util.Vector;

/**
 * 将文件转换为流 file://c:/dic.txt
 * 
 * @author ansj
 *
 */
public class File2Stream extends PathToStream {

    private static final Log LOG = LogFactory.getLog(File2Stream.class);

    @Override
    public InputStream toStream(String path) {
        LOG.info("path to stream " + path);

        if (path.startsWith("file://")) {
            path = path.substring(7);
        }

        File file = new File(path);

        if (file.exists() && file.canRead()) {

            try {
                if (file.isDirectory()) {
                    return multiple(path);
                } else {
                    return new FileInputStream(file);
                }
            } catch (Exception e) {
                throw new LibraryException(e);
            }
        }
        throw new LibraryException(
                        " path :" + path + " file:" + file.getAbsolutePath() + " not found or can not to read");

    }

    private InputStream multiple(String path) throws FileNotFoundException {
        File[] libs = new File[0];

        File file = new File(path);

        if (file.exists() && file.canRead()) {
            if (file.isFile()) {
                libs = new File[1];
                libs[0] = file;
            } else if (file.isDirectory()) {

                File[] files = file.listFiles(new FileFilter() {
                    @Override
                    public boolean accept(File file) {
                        return file.canRead() && !file.isHidden() && !file.isDirectory();
                    }
                });

                if (files != null && files.length > 0) {
                    libs = files;
                }
            }
        }

        if (libs.length == 0) {
            throw new LibraryException("not find any file in path : " + path);
        }

        if (libs.length == 1) {
            return new FileInputStream(libs[0]);
        }

        Vector<InputStream> vector = new Vector<>(libs.length);

        for (int i = 0; i < libs.length; i++) {
            vector.add(new FileInputStream(libs[i]));
        }

        return new SequenceInputStream(vector.elements());
    }

}
