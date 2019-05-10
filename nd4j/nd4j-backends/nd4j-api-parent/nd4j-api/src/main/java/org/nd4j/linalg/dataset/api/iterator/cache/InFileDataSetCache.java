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

package org.nd4j.linalg.dataset.api.iterator.cache;

import org.nd4j.linalg.dataset.DataSet;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;

/**
 * Created by anton on 7/18/16.
 */
public class InFileDataSetCache implements DataSetCache {
    private File cacheDirectory;

    public InFileDataSetCache(File cacheDirectory) {
        if (cacheDirectory.exists() && !cacheDirectory.isDirectory()) {
            throw new IllegalArgumentException("can't use path " + cacheDirectory + " as file cache directory "
                            + "because it already exists, but is not a directory");
        }
        this.cacheDirectory = cacheDirectory;
    }

    public InFileDataSetCache(Path cacheDirectory) {
        this(cacheDirectory.toFile());
    }

    public InFileDataSetCache(String cacheDirectory) {
        this(new File(cacheDirectory));
    }

    private File resolveKey(String key) {
        String filename = key.replaceAll("[^a-zA-Z0-9.-]", "_");
        return new File(cacheDirectory, filename);
    }

    private File namespaceFile(String namespace) {
        String filename = String.format("%s-complete.txt", namespace);
        return new File(cacheDirectory, filename);
    }

    @Override
    public boolean isComplete(String namespace) {
        return namespaceFile(namespace).exists();
    }

    @Override
    public void setComplete(String namespace, boolean value) {
        File file = namespaceFile(namespace);
        if (value) {
            if (!file.exists()) {
                File parentFile = file.getParentFile();
                parentFile.mkdirs();
                try {
                    file.createNewFile();
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
        } else {
            if (file.exists()) {
                file.delete();
            }
        }
    }

    @Override
    public DataSet get(String key) {
        File file = resolveKey(key);

        if (!file.exists()) {
            return null;
        } else if (!file.isFile()) {
            throw new IllegalStateException("ERROR: cannot read DataSet: cache path " + file + " is not a file");
        } else {
            DataSet ds = new DataSet();
            ds.load(file);
            return ds;
        }
    }

    @Override
    public void put(String key, DataSet dataSet) {
        File file = resolveKey(key);

        File parentDir = file.getParentFile();
        if (!parentDir.exists()) {
            if (!parentDir.mkdirs()) {
                throw new IllegalStateException("ERROR: cannot create parent directory: " + parentDir);
            }
        }

        if (file.exists()) {
            file.delete();
        }

        dataSet.save(file);
    }

    @Override
    public boolean contains(String key) {
        File file = resolveKey(key);

        Boolean exists = file.exists();
        if (exists && !file.isFile()) {
            throw new IllegalStateException("ERROR: DataSet cache path " + file + " exists but is not a file");
        }

        return exists;
    }
}
