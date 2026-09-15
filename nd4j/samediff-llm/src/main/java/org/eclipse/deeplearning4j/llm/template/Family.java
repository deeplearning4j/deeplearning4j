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
 *  *  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  *  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  *  License for the specific language governing permissions and limitations
 *  *  under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.llm.template;

/**
 * Architecture families with dedicated chat templates in this package.
 *
 * @author Eclipse Deeplearning4j Contributors
 */
public enum Family {
    /** Qwen3.5 / Qwen3.8 (incl. Flash-Next) ChatML with &lt;think&gt; blocks. */
    QWEN,
    /** GLM-5.x (zai-org) square-bracket roles with reasoning-effort system block. */
    GLM,
    /** DeepSeek-V4.1 full-width-bar prompt encoder format. */
    DEEPSEEK
}
