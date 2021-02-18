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
package org.datavec.image;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.common.tests.AbstractAssertTestsClass;
import org.nd4j.common.tests.BaseND4JTest;

import java.util.*;

@Slf4j
public class AssertTestsExtendBaseClass extends AbstractAssertTestsClass {

	@Override
	protected Set<Class<?>> getExclusions() {
		//Set of classes that are exclusions to the rule (either run manually or have their own logging + timeouts)
		return new HashSet<>();
	}

	@Override
	protected String getPackageName() {
		return "org.datavec.image";
	}

	@Override
	protected Class<?> getBaseClass() {
		return BaseND4JTest.class;
	}
}
