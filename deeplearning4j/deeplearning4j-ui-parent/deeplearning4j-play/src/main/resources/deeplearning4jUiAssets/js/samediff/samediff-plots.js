/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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


function renderLineChart(/*jquery selector*/ element, label, xDataArray, yDataArray ){


    var toPlot = [];
    for(var i=0; i<xDataArray.length; i++ ){
        toPlot.push([xDataArray[i], yDataArray[i]]);
    }

    element.unbind();

    var yMax = Math.max.apply(Math, yDataArray);
    var yMin = Math.min.apply(Math, yDataArray);
    if(yMin > 0){
        yMin = 0.0;
    }

    var plotOptions = {
        series: {
            lines: {
                show: true,
                lineWidth: 2
            }
        },
        grid: {
            hoverable: true,
            clickable: true,
            tickColor: "#dddddd",
            borderWidth: 0
        },
        yaxis: {min: yMin, max: yMax},
        colors: ["#FA5833","rgba(65,182,240,0.3)","#000000"],
        selection: {
            mode: "x"
        }
    };

    var plotData = [{data: toPlot, label: label}];
    var plot = $.plot(element, plotData, plotOptions);

}

