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

package org.deeplearning4j.arbiter.server.cli;

import com.beust.jcommander.IParameterValidator;
import com.beust.jcommander.ParameterException;
import org.deeplearning4j.arbiter.server.ArbiterCliRunner;

/**
 * Created by agibsonccc on 3/13/17.
 */
public class NeuralNetTypeValidator implements IParameterValidator {
    /**
     * Validate the parameter.
     *
     * @param name  The name of the parameter (e.g. "-host").
     * @param value The value of the parameter that we need to validate
     * @throws ParameterException Thrown if the value of the parameter is invalid.
     */
    @Override
    public void validate(String name, String value) throws ParameterException {
        if(!value.equals(ArbiterCliRunner.MULTI_LAYER_NETWORK) || value.equals(ArbiterCliRunner.COMP_GRAPH)) {
            throw new ParameterException("Neural net type can only be " + ArbiterCliRunner.COMP_GRAPH + " or " + ArbiterCliRunner.MULTI_LAYER_NETWORK);

        }
    }
}
