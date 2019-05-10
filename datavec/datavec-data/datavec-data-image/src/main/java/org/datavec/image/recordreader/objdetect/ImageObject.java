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

package org.datavec.image.recordreader.objdetect;

import lombok.Data;

@Data
public class ImageObject {

    private final int x1;
    private final int y1;
    private final int x2;
    private final int y2;
    private final String label;

    public ImageObject(int x1, int y1, int x2, int y2, String label){
        if(x1 > x2 || y1 > y2){
            throw new IllegalArgumentException("Invalid input: (x1,y1), top left position must have values less than" +
                    " (x2,y2) bottom right position. Got: (" + x1 + "," + y1 + "), (" + x2 + "," + y2 + ")");
        }

        this.x1 = x1;
        this.y1 = y1;
        this.x2 = x2;
        this.y2 = y2;
        this.label = label;
    }

    public double getXCenterPixels(){
        return (x1 + x2) / 2.0;
    }

    public double getYCenterPixels(){
        return (y1 + y2) / 2.0;
    }
}
