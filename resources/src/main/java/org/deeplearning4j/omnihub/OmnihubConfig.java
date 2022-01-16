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
package org.deeplearning4j.omnihub;

import java.io.File;

public class OmnihubConfig {

    public final static String OMNIHUB_HOME = "OMNIHUB_HOME";
    public final static String OMNIHUB_URL = "OMNIHUB_URL";
    public static final String DEFAULT_OMNIHUB_URL = "https://raw.githubusercontent.com/KonduitAI/omnihub-zoo/main";

    /**
     * Return the omnihub hurl defaulting to
     * {@link #DEFAULT_OMNIHUB_URL}
     * if the {@link #OMNIHUB_URL} is not specified.
     * @return
     */
    public static String getOmnihubUrl() {
        if(System.getenv(OMNIHUB_URL) != null) {
            return System.getenv(OMNIHUB_URL);
        } else {
            return DEFAULT_OMNIHUB_URL;
        }
    }

    /**
     * return the default omnihub home at $USER/.omnihub or
     * value of the environment variable {@link #OMNIHUB_HOME} if applicable
     * @return
     */
    public static File getOmnihubHome() {
        if(System.getenv(OMNIHUB_HOME) != null) {
            return new File(OMNIHUB_HOME);
        } else {
            return new File(System.getProperty("user.home"),".omnihub");
        }
    }
}
