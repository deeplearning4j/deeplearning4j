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

package org.deeplearning4j.ui.i18n;

import org.deeplearning4j.ui.api.I18N;

/**
 * Returns the currently used I18N (Internationalization) class
 *
 * @author Alex Black
 */
public class I18NProvider {

    private static I18N i18n = DefaultI18N.getInstance();

    /**
     * Get the current/global I18N instance
     */
    public static I18N getInstance() {
        return i18n;
    }

}
