/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.arbiter.optimize.api;


import org.arbiter.nn.api.Model;

import java.io.Serializable;

/**
 * Each epoch the listener is called, mainly used for debugging or visualizations
 * @author Adam Gibson
 *
 */
public interface IterationListener extends Serializable {
	/**
	 * Event listener for each iteration
	 * @param iteration the iteration
     * @param model the model iterating
	 */
	void iterationDone(Model model,int iteration);
	
}
