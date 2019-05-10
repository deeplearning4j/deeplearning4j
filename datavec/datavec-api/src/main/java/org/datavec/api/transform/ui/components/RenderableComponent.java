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

package org.datavec.api.transform.ui.components;

import lombok.Data;
import org.nd4j.shade.jackson.annotation.JsonSubTypes;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.WRAPPER_OBJECT)
@JsonSubTypes(value = {
                @JsonSubTypes.Type(value = RenderableComponentLineChart.class, name = "RenderableComponentLineChart"),
                @JsonSubTypes.Type(value = RenderableComponentTable.class, name = "RenderableComponentTable"),
                @JsonSubTypes.Type(value = RenderableComponentHistogram.class, name = "RenderableComponentHistogram")})
@Data
public abstract class RenderableComponent {

    /** Component type: used by the Arbiter UI to determine how to decode and render the object which is
     * represented by the JSON representation of this object*/
    protected final String componentType;

    public RenderableComponent(String componentType) {
        this.componentType = componentType;
    }

}
