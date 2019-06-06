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

package org.deeplearning4j.datasets.fetchers;

import org.deeplearning4j.BaseDL4JTest;
import org.junit.Test;

import java.io.File;

import static org.junit.Assert.assertTrue;

/**
 * @author saudet
 */
public class SvhnDataFetcherTest extends BaseDL4JTest {

    @Test
    public void testSvhnDataFetcher() throws Exception {
        SvhnDataFetcher fetch = new SvhnDataFetcher();
        File path = fetch.getDataSetPath(DataSetType.TRAIN);
        File path2 = fetch.getDataSetPath(DataSetType.TEST);
        File path3 = fetch.getDataSetPath(DataSetType.VALIDATION);

        assertTrue(path.isDirectory());
        assertTrue(path2.isDirectory());
        assertTrue(path3.isDirectory());
    }
}
