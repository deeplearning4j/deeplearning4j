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

package org.deeplearning4j;

/**
 * A pipeline consists of a
 * set of input,output, and datavec uris.
 * This is used to build a composable data pipeline
 * that can process or integrate any kind of data
 *
 * @author Adam Gibson
 */
public interface Pipeline {
    /**
     * Origin data
     * @return the input destination uris
     */
    String[] inputUris();

    /**
     * Output destinations
     * @return the output destinations
     */
    String[] outputUris();

    /**
     * The datavec uris to use
     * @return the uris used for datavec
     * vectorization
     */
    String[] datavecUris();

}
