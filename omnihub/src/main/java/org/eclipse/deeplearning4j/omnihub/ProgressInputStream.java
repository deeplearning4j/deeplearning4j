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
package org.eclipse.deeplearning4j.omnihub;

import me.tongfei.progressbar.ProgressBar;
import org.apache.commons.io.input.CountingInputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;

public class ProgressInputStream extends CountingInputStream {

    private long max;
    private ProgressBar pb = new ProgressBar("Download model:", 100); // name, initial max
    public ProgressInputStream(InputStream in,long max) {
        super(in);
        this.max = max;
        pb.start();
    }

    @Override
    protected synchronized void afterRead(int n) {
        super.afterRead(n);
        double progress = (getByteCount() / (double) max) * 100.0;
        pb.stepTo((long) progress);
    }


    @Override
    public void close() throws IOException {
        super.close();
        pb.stop();
    }
}
