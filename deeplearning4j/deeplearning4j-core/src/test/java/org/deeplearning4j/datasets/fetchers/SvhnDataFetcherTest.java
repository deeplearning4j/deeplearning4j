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
package org.deeplearning4j.datasets.fetchers;

import org.deeplearning4j.BaseDL4JTest;
import org.junit.Rule;
import org.junit.jupiter.api.Test;
import org.junit.rules.Timeout;
import java.io.File;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.extension.ExtendWith;

/**
 * @author saudet
 */
@DisplayName("Svhn Data Fetcher Test")
class SvhnDataFetcherTest extends BaseDL4JTest {

    @Override
    public long getTimeoutMilliseconds() {
        // Shouldn't take this long but slow download or drive access on CI machines may need extra time.
        return 480_000_000L;
    }

    @Test
    @DisplayName("Test Svhn Data Fetcher")
    void testSvhnDataFetcher() throws Exception {
        // Ignore unless integration tests - CI can get caught up on slow disk access
        assumeTrue(isIntegrationTests());
        SvhnDataFetcher fetch = new SvhnDataFetcher();
        File path = fetch.getDataSetPath(DataSetType.TRAIN);
        File path2 = fetch.getDataSetPath(DataSetType.TEST);
        File path3 = fetch.getDataSetPath(DataSetType.VALIDATION);
        assertTrue(path.isDirectory());
        assertTrue(path2.isDirectory());
        assertTrue(path3.isDirectory());
    }
}
