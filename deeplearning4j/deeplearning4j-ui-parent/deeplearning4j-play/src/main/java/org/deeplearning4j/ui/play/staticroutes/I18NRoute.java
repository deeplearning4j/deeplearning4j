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

package org.deeplearning4j.ui.play.staticroutes;

import org.deeplearning4j.ui.i18n.I18NProvider;
import play.mvc.Result;

import java.util.function.Function;

import static play.mvc.Results.ok;

/**
 * Route for global internationalization setting
 *
 * @author Alex Black
 */
public class I18NRoute implements Function<String, Result> {
    @Override
    public Result apply(String s) {
        I18NProvider.getInstance().setDefaultLanguage(s);
        return ok();
    }
}
