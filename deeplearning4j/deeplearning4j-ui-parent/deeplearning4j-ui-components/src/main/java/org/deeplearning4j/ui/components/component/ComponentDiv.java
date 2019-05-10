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

package org.deeplearning4j.ui.components.component;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.ui.api.Component;
import org.deeplearning4j.ui.api.Style;
import org.nd4j.shade.jackson.annotation.JsonInclude;

import java.util.Collection;

/**
 * Div component (as in, HTML div)
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(callSuper = true)
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ComponentDiv extends Component {
    public static final String COMPONENT_TYPE = "ComponentDiv";

    private Component[] components;

    public ComponentDiv() {
        super(COMPONENT_TYPE, null);
    }


    public ComponentDiv(Style style, Component... components) {
        super(COMPONENT_TYPE, style);
        this.components = components;
    }

    public ComponentDiv(Style style, Collection<Component> componentCollection) {
        this(style, (componentCollection == null ? null
                        : componentCollection.toArray(new Component[componentCollection.size()])));
    }
}
