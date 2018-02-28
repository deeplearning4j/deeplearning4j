/*-
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.api.util;

import org.nd4j.util.StringUtils;

import java.util.concurrent.TimeUnit;

/**
 * @deprecated Use {@link org.nd4j.util.StringUtils}
 */
@Deprecated
public class TransformUtils {

    /**
     * @deprecated Use {@link StringUtils#timeUnitToString(long, TimeUnit)}
     */
    @Deprecated
    public static String timeUnitToString(long time, TimeUnit unit) {
        return StringUtils.timeUnitToString(time, unit);
    }

    /**
     * @deprecated Use {@link StringUtils#stringToTimeUnit(String)}
     */
    @Deprecated
    public static TimeUnit stringToTimeUnit(String str) {
        return StringUtils.stringToTimeUnit(str);
    }
}
