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

package vizdoom;

public class Label{
    public int objectId;
    public String objectName;
    public byte value;
    public double objectPositionX;
    public double objectPositionY;
    public double objectPositionZ;

    Label(int id, String name, byte value, double positionX, double positionY, double positionZ){
        this.objectId = objectId;
        this.objectName = objectName;
        this.value = value;
        this.objectPositionX = positionX;
        this.objectPositionY = positionY;
        this.objectPositionZ = positionZ;
    }
}
