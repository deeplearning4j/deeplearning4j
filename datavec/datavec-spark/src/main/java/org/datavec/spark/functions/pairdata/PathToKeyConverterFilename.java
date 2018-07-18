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

package org.datavec.spark.functions.pairdata;

import org.apache.commons.io.FilenameUtils;

/** Convert the path to a key by taking the full file name (excluding the file extension and directories) */
public class PathToKeyConverterFilename implements PathToKeyConverter {

    @Override
    public String getKey(String path) {
        return FilenameUtils.getBaseName(path);
    }
}
