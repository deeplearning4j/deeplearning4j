/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.nd4j.autodiff.samediff;

/**
 * Internal interface used to apply a transform to any arguments used within a certain block
 *
 * Intended for internal use only.
 *
 * Managed with {@link SameDiff#addArgumentInterceptor(ArgumentInterceptor)}, {@link SameDiff#removeArgumentInterceptor()},
 * {@link SameDiff#pauseArgumentInterceptor()}, and {@link SameDiff#unpauseArgumentInterceptor()}
 *
 */
public interface ArgumentInterceptor {
    SDVariable intercept(SDVariable argument);
}
