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

package org.deeplearning4j.ui.storage;

import org.deeplearning4j.ui.storage.mapdb.MapDBStatsStorage;

import java.io.File;

/**
 * A StatsStorage implementation that stores UI data in a file for persistence.<br>
 * Can be used for multiple instances, and across multiple independent runs. Data can be loaded later in a separate
 * JVM instance by passing the same file location to both.<br>
 * Internally, uses {@link MapDBStatsStorage}
 *
 * @author Alex Black
 */
public class FileStatsStorage extends MapDBStatsStorage {

    private final File file;

    public FileStatsStorage(File f) {
        super(f);
        this.file = f;
    }

    @Override
    public String toString() {
        return "FileStatsStorage(" + file.getPath() + ")";
    }
}
