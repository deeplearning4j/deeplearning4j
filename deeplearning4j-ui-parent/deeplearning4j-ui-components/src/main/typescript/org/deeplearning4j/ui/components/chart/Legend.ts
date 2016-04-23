/*
 *
 *  * Copyright 2016 Skymind,Inc.
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
 *
 */

class Legend {

    //TODO: make these configurable...
    private static offsetX: number = 15;
    private static offsetY: number = 15;
    private static padding: number = 8;
    private static separation: number = 12;
    private static boxSize: number = 10;
    private static fillColor: string = "#FFFFFF";
    private static legendOpacity: number = 0.75;
    private static borderStrokeColor: string = "#000000";


    static legendFn = (function(g: any) {
        //Get SVG and legend box/items:
        var svg = d3.select(g.property("nearestViewportElement"));
        var legendBox = g.selectAll(".outerRect").data([true]);
        var legendItems = g.selectAll(".legendElement").data([true]);

        legendBox.enter().append("rect").attr("class","outerRect");
        legendItems.enter().append("g").attr("class","legendElement");

        var legendElements: any[] = [];
        svg.selectAll("[data-legend]").each(function() {
            var thisVar = d3.select(this);
            legendElements.push({
                label: thisVar.attr("data-legend"),
                color: thisVar.style("fill")
            });
        });


        //Add rectangles for color
        legendItems.selectAll("rect")
            .data(legendElements,function(d) { return d.label})
            .call(function(d) { d.enter().append("rect")})
            .call(function(d) { d.exit().remove()})
            .attr("x",0)
            .attr("y",function(d,i) { return i*Legend.separation-Legend.boxSize+"px"})
            .attr("width",Legend.boxSize)
            .attr("height",Legend.boxSize)
            //.style("fill",function(d) { return d.value.color});
            .style("fill",function(d) { return d.color});

        //Add labels
        legendItems.selectAll("text")
            .data(legendElements,function(d) { return d.label})
            .call(function(d) { d.enter().append("text")})
            .call(function(d) { d.exit().remove()})
            .attr("y",function(d,i) { return i*Legend.separation + "px"})
            .attr("x",(Legend.padding + Legend.boxSize) + "px")
            .text(function(d) { return d.label});

        //Add the outer box
        var legendBoundingBox: any = legendItems[0][0].getBBox();
        legendBox.attr("x",(legendBoundingBox.x-Legend.padding))
            .attr("y",(legendBoundingBox.y-Legend.padding))
            .attr("height",(legendBoundingBox.height+2*Legend.padding))
            .attr("width",(legendBoundingBox.width+2*Legend.padding))
            .style("fill",Legend.fillColor)
            .style("stroke",Legend.borderStrokeColor)
            .style("opacity",Legend.legendOpacity);

        svg.selectAll(".legend").attr("transform","translate(" + Legend.offsetX + "," + Legend.offsetY + ")");
    });
}