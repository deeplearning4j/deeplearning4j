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

package org.deeplearning4j.datasets;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.base.MnistFetcher;
import org.deeplearning4j.datasets.fetchers.Cifar10Fetcher;
import org.deeplearning4j.datasets.fetchers.TinyImageNetFetcher;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;

import java.util.concurrent.*;

import static org.junit.jupiter.api.Assertions.assertNull;

@NativeTag
@Tag(TagNames.FILE_IO)
public class ConcurrentDownloaderTest extends BaseDL4JTest {

    @Override
    public long getTimeoutMilliseconds() {
        return 180000L;
    }

    @Test
    public void testConcurrentDownloadingOfSameDataSet() throws Exception {
        final ExecutorService executorService = Executors.newFixedThreadPool(24);
        final ExecutorCompletionService<Object> completionService = new ExecutorCompletionService<>(executorService);

        final int expected = 24;
        for (int i = 0; i < expected; i++) {
            completionService.submit(() -> (new MnistFetcher()).downloadAndUntar());
        }

        int received = 0;
        Throwable e = null;
        while(received < expected && e == null){
            final Future<Object> result = completionService.take();
            try{
                result.get();
                received++;
            }catch (ExecutionException executionException) {
                e = executionException.getCause();
            }
        }

        executorService.shutdownNow();
        assertNull(e);
    }
}
