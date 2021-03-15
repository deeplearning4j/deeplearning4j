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

package org.deeplearning4j.ui.api;

import java.util.Map;

public interface I18N {

    /**
     * Get the specified message in the default language (according to {@link #getDefaultLanguage()}
     *
     * @param key Key value
     * @return Message for the given key, or null if none is found/available
     */
    String getMessage(String key);

    /**
     * Get the specified message for the specified language
     *
     * @param langCode ISO 639-1 language code: "en", "ja", etc
     * @param key      Key value for the message to retrieve
     * @return Message for the given key/language pair, or null if none is found
     */
    String getMessage(String langCode, String key);

    /**
     * Get the currently set default language as an ISO 639-1 code
     *
     * @return The current default language
     */
    String getDefaultLanguage();

    /**
     * Set the default language
     *
     * @param langCode Language code, as an ISO 639-1 code
     */
    void setDefaultLanguage(String langCode);

    /**
     * Get all internationalization messages for the specified language code
     * @param langCode Language code
     * @return Messages
     */
    Map<String,String> getMessages(String langCode);

}
