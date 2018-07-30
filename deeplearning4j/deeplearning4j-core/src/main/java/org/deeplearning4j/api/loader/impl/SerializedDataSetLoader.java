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

package org.deeplearning4j.api.loader.impl;

import org.deeplearning4j.api.loader.DataSetLoader;
import org.nd4j.api.loader.Source;
import org.nd4j.linalg.dataset.DataSet;

import java.io.IOException;
import java.io.InputStream;

/**
 * Loads DataSets using {@link DataSet#load(InputStream)}
 *
 * @author Alex Black
 */
public class SerializedDataSetLoader implements DataSetLoader {
    @Override
    public DataSet load(Source source) throws IOException {
        DataSet ds = new DataSet();
        try(InputStream is = source.getInputStream()){
            ds.load(is);
        }
        return ds;
    }
}
