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

function Layers() {
    this.layers = [];
    this.totalLayers = 0;
    this.maximumX = 0;
    this.maximumY = 0;

    this.getLayerForID = function(id) {
        for (var i = 0; i < this.layers.length; i++) {
            if (this.layers[i].id == id) {
                return this.layers[i];
            }
        };
        return null;
    }

    this.attach = function( layer) {
        this.layers.push(layer);
        this.totalLayers++;
        if (this.maximumX < layer.x) this.maximumX = layer.x;
        if (this.maximumY < layer.y) this.maximumY = layer.y;
    }

    this.getLayersForY = function(y) {
        var result = [];
            for (var i = 0; i < this.layers.length; i++) {
                if (this.layers[i].y == y) {
                    result.push(this.layers[i]);
                }
            }
         return result;
    }

    this.getLayersForX = function (x) {
        var result = [];
        for (var i = 0; i < this.layers.length; i++) {
            if (this.layers[i].x == x) {
                result.push(this.layers[i]);
            }
        }
        return result;
    }

    this.getLayerForXY = function (x, y) {
        var layerX = this.getLayersForX(x);
        for (var i = 0; i < layerX.length; i++) {
            if (layerX[i].y == y) return layerX[i];
        }
        return null;
    }

    this.getLayerForYX = function (x, y) {
            var layerY = this.getLayersForY(y);
            for (var i = 0; i < layerY.length; i++) {
                if (layerY[i].x == x) return layerY[i];
            }
            return null;
        }
}