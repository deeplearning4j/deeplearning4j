/*
    This is going to be proper version of WeightsRender.js
    TODO list:
    1. Initialize charts once, and just update them with new data
    2. Add 3D surface as overall view for gradients and weights
*/

    var modelSelector = new Array();
    var magnitudesSelector = new Array();

    // current visible chart, all other charts will be hidden
    var visibleModel = "";
    var visibleGradient = "";
    var visibleMagnitude = "";


    // all charts have equal size
    var margin = {top: 10, right: 30, bottom: 30, left: 30},
    var width = 650 - margin.left - margin.right,
    var height = 350 - margin.top - margin.bottom;

    // we''ll define every single data source as global object, and we'll them as references for D3 charts
    var gScore = new Array();
    var gModel = new Array();
    var gGradient = new Array();
    var gMagnitude = new Array();

    // we must ensure, that charts initialized only once
    var isInit = false;

    var contains = function(needle) {
        // Per spec, the way to identify NaN is that it is not equal to itself
        var findNaN = needle !== needle;
        var indexOf;

        if(!findNaN && typeof Array.prototype.indexOf === 'function') {
            indexOf = Array.prototype.indexOf;
        } else {
            indexOf = function(needle) {
                var i = -1, index = -1;

                for(i = 0; i < this.length; i++) {
                    var item = this[i];

                    if((findNaN && item !== item) || item === needle) {
                        index = i;
                        break;
                    }
                }

                return index;
            };
        }

        return indexOf.call(this, needle) > -1;
    };


    function appendHistogram(values,selector) {
        // A formatter for counts.
        var formatCount = d3.format(",.0f");


        var data = values;
        var min = d3.min(data);
        var max = d3.max(data);
        if(isNaN(min)){
            min = 0.0;
            max = 1.0;
        }

        var x = d3.scale.linear()
                .domain([min, max])
                .range([0, width]);

        // Generate a histogram using twenty uniformly-spaced bins.
        var data = d3.layout.histogram()
                .bins(x.ticks(20))
                (values);

        var y = d3.scale.linear()
                .domain([0, d3.max(data, function(d) { return d.y; })])
                .range([height, 0]);

        var xAxis = d3.svg.axis()
                .scale(x)
                .orient("bottom");

        var svg = d3.select(selector).append("svg")
                .attr("width", width + margin.left + margin.right)
                .attr("height", height + margin.top + margin.bottom)
                .append("g")
                .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

        var bar = svg.selectAll(".bar")
                .data(data)
                .enter().append("g")
                .attr("class", "bar")
                .attr("transform", function(d) { return "translate(" + x(d.x) + "," + y(d.y) + ")"; });

        bar.append("rect")
                .attr("x", 1)
                .attr("width", x(min+data[0].dx) -1 )
                .attr("height", function(d) { return height - y(d.y); });

        bar.append("text")
                .attr("dy", ".75em")
                .attr("y", 6)
                .attr("x", x(min+data[0].dx) / 2)
                .attr("text-anchor", "middle")
                .text(function(d) { return formatCount(d.y); });

        svg.append("g")
                .attr("class", "x axis")
                .attr("transform", "translate(0," + height + ")")
                .call(xAxis);
    }

 function appendLineChart(values,selector){
        // Set the dimensions of the canvas / graph

        // Set the ranges
        var x = d3.scale.linear().range([0, width]);
        var y = d3.scale.linear().range([height, 0]);

        // Define the axes
        var xAxis = d3.svg.axis().scale(x)
                .innerTickSize(-height)     //used as grid line
                .orient("bottom").ticks(5);

        var yAxis = d3.svg.axis().scale(y)
                .innerTickSize(-width)      //used as grid line
                .orient("left").ticks(5);

        // Define the line
        var valueline = d3.svg.line()
                .x(function(d,i) { return x(i); })
                .y(function(d) { return y(d); });

        // Adds the svg canvas
        var svg = d3.select(selector)
                .append("svg")
                .attr("width", width + margin.left + margin.right)
                .attr("height", height + margin.top + margin.bottom)
                .append("g")
                .attr("transform",
                "translate(" + margin.left + "," + margin.top + ")");

        // Scale the range of the data
        var max = d3.max(values);
        x.domain([0,values.length]);
        y.domain([0, max]);

        // Add the valueline path.
        svg.append("path")
                .attr("class", "line")
                .attr("d", valueline(values));

        // Add the X Axis
        svg.append("g")
                .attr("class", "x axis")
                .attr("transform", "translate(0," + height + ")")
                .call(xAxis);

        // Add the Y Axis
        svg.append("g")
                .attr("class", "y axis")
                .call(yAxis);
    }

    function appendMultiLineChart(map,selector){
        var keys = Object.keys(map)
        if(keys.length == 0) return;    //nothing to plot


        // Set the ranges
        var x = d3.scale.linear().range([0, width]);
        var y = d3.scale.linear().range([height, 0]);

        // Define the axes
        var xAxis = d3.svg.axis().scale(x)
                .innerTickSize(-height)     //used as grid line
                .orient("bottom").ticks(5);

        var yAxis = d3.svg.axis().scale(y)
                .innerTickSize(-width)      //used as grid line
                .orient("left").ticks(5);

        // Define the line
        var valueline = d3.svg.line()
                .x(function(d,i) { return x(i); })
                .y(function(d) { return y(d); });

        // Adds the svg canvas
        var svg = d3.select(selector)
                .append("svg")
                .attr("width", width + margin.left + margin.right)
                .attr("height", height + margin.top + margin.bottom)
                .append("g")
                .attr("transform",
                "translate(" + margin.left + "," + margin.top + ")");

        // Scale the range of the data
        var max = -Number.MAX_VALUE;
        var size = 1;
        for( var key in map ){
            var values = map[key];
            var thisMax = d3.max(values);
            if(thisMax > max) max = thisMax;
            size = values.length;
        }
        x.domain([0,size]);
        y.domain([0, max]);

        // Add the valueline path.
        var color = d3.scale.category10();
        var i=0;
        for( var key in map ){
            var values = map[key];
            svg.append("path")
                .attr("class", "line")
                .style("stroke", color(i))
                .attr("d", valueline(values));
            i++;
        }

        // Add the X Axis
        svg.append("g")
                .attr("class", "x axis")
                .attr("transform", "translate(0," + height + ")")
                .call(xAxis);

        // Add the Y Axis
        svg.append("g")
                .attr("class", "y axis")
                .call(yAxis);

        //Add legend
        var legendSpace = width/i;
        i = 0;
        for( var key in map ){
            var values = map[key];
            var last = values[values.length-1];
            var toDisplay = key + " (" + last.toPrecision(5) + ") ";
            svg.append("text")
                .attr("x", (legendSpace/2)+i*legendSpace) // spacing
                .attr("y", height + (margin.bottom/2)+ 5)
                .attr("class", "legend")    // style the legend
                .style("fill", color(i))
                .text(toDisplay);

            i++;
        }
    }

var timed = function() {
                    $.ajax({
                        url:"${path}" + "/updated",
                        async: false,
                        success: function( data ) {
                                    /*
                                        /weights/data should be changed to /weights/data/{time} and only delta should be passed over network
                                    */
                                    d3.json("${path}"+'/data',function(error,json) {

                                        //Get last update time; do nothing if not a new update
                                        var updateTime = json['lastUpdateTime'];
                                        var lastUpdateTime = $('#lastupdate .updatetime').text();
                                        if(updateTime == lastUpdateTime) return;


                                        var model = json['parameters'];
                                        var gradient = json['gradients'];
                                        var score = json['score'];
                                        var scores = json['scores'];
                                        var updateMagnitudes = json['updateMagnitudes'];
                                        var paramMagnitudes = json['paramMagnitudes'];

                                        if(!model || !gradient || !score || !scores || !updateMagnitudes || !paramMagnitudes ) {
                                            setTimeout(timed, 10000);
                                            return;
                                        }
                                        $('.score').html('' + score);


                                        $('#scores .chart').html('');
                                        var scdiv = '<div class="scorechart"></div>';
                                        $('#scores .chart').append(scdiv);
                                        appendLineChart(scores,'#scores .chart');

                                        //clear out body of where the chart content will go
                                        $('#model .charts').html('');
                                        $('#gradient .charts').html('');
                                        var keys = Object.keys(model);
                                        for(var i = 0; i < keys.length; i++) {
                                            var key = keys[i];
                                            //model id class charts
                                            var selectorModel = '#model .charts';
                                            var selectorGradient = '#gradient .charts';
                                            //append div to each node where the chart content will
                                            //go and pass that in to the chart renderer
                                            var divModel = '<div id="model'+ key+'" class="' + key + '" style="' +  ((visibleModel == key) ? "visibility: visible; display: block;" : "visibility: hidden; display: none;") +';"></div>';
                                            var divGradient = '<div id="gradient'+ key+'" class="' + key + '" style="' +  ((visibleGradient == key) ? "visibility: visible; display: block;" : "visibility: hidden; display: none;") +';"></div>';
                                            $(selectorModel).append(divModel);
                                            $(selectorGradient).append(divGradient);
                                            appendHistogram(model[key]['dataBuffer'],selectorModel + ' .' + key);
                                            appendHistogram(gradient[key]['dataBuffer'],selectorGradient + ' .' + key);
                                            /*
                                                update selector box if needed
                                            */
                                            if (!contains.call(modelSelector, key)) {
                                                console.log("Adding model selector: " + key);
                                                modelSelector.push(key);

                                                $("#modelSelector").append("<option value='"+ key+"'>" + key + "</option>");
                                                $("#gradientSelector").append("<option value='"+ key+"'>" + key + "</option>");
                                            }
                                        }


                                        //Plot mean magnitudes: weights and params
                                        $('#magnitudes .charts').html('');
                                        for(var i=0; i<updateMagnitudes.length; i++ ){
                                            //Maps:
                                            var mapParams = paramMagnitudes[i];
                                            var mapUpdates = updateMagnitudes[i];

                                            var selectorModel = '#magnitudes .charts'
                                            var div = '<div id="layer' + i + 'param" class="layer' + i + 'param" style="' +  ((visibleMagnitude == "layer" + i + "param" ) ? "visibility: visible; display: block;" : "visibility: hidden; display: none;") +';"></div>';
                                            $(selectorModel).append(div);
                                            appendMultiLineChart(mapParams,selectorModel + ' .layer' + i + 'param');
                                            div = '<div id="layer' + i + 'grad" class="layer' + i + 'grad" style="' +  ((visibleMagnitude == "layer" + i + "grad" ) ? "visibility: visible; display: block;" : "visibility: hidden; display: none;") +';"></div>';
                                            $(selectorModel).append(div);
                                            appendMultiLineChart(mapUpdates,selectorModel + ' .layer' + i + 'grad');

                                            if (!contains.call(magnitudesSelector, key)) {
                                                console.log("Adding magnitudes selector: " + key);
                                                magnitudesSelector.push(key);

                                                $("#magnitudeSelector").append("<option value='layer" + i + "param'>Layer " + i + " Parameter Mean Magnitudes</option>");
                                                $("#magnitudeSelector").append("<option value='layer" + i + "grad'>Layer " + i + " Update/Gradient Mean Magnitudes</option>");
                                            }
                                        }

                                        // this hack allows first selection of visible model histo
                                        if (visibleModel == "") {
                                            $("#modelSelector").val($("#modelSelector option:first").val());
                                            selectModel();
                                        }

                                        if (visibleGradient == "") {
                                            $("#gradientSelector").val($("#gradientSelector option:first").val());
                                            selectGradient();
                                        }

                                        if (visibleMagnitude == "") {
                                            $("#magnitudeSelector").val($("#magnitudeSelector option:first").val());
                                            selectMagnitude();
                                        }

                                        var time = new Date(updateTime);
                                        $('#updatetime').html(time.customFormat("#DD#/#MM#/#YYYY# #hhh#:#mm#:#ss#"));

                                        // all subsequent refreshes are delayed by 10 seconds
                                        setTimeout(timed, 10000)
                                    });
                    }
                })
            };

// first update is fired almost immediately, 2s timeout
setTimeout(timed,2000);


    function selectModel() {
        console.log("Switching off model view: " + visibleModel);
        if (visibleModel != "") {
            $("#model" + visibleModel).css("visibility","hidden");
            $("#model" + visibleModel).css("display","none");
        }

        visibleModel = $("#modelSelector").val();
        $("#model" + visibleModel).css("visibility","visible");
        $("#model" + visibleModel).css("display","block");
        console.log("Switching on model view:" + visibleModel);
    }

    function selectGradient() {
            console.log("Switching off gradient view: " + visibleGradient);
            if (visibleGradient != "") {
                $("#gradient" + visibleGradient).css("visibility","hidden");
                $("#gradient" + visibleGradient).css("display","none");
            }

            visibleGradient = $("#gradientSelector").val();
            $("#gradient" + visibleGradient).css("visibility","visible");
            $("#gradient" + visibleGradient).css("display","block");
            console.log("Switching on gradient view:" + visibleGradient);
    }

    function selectMagnitude() {
        console.log("Switching off magnitude view: " + visibleMagnitude);
        if (visibleMagnitude != "") {
            $("#" + visibleMagnitude).css("visibility","hidden");
            $("#" + visibleMagnitude).css("display","none");
        }

        visibleMagnitude = $("#magnitudeSelector").val();

        $("#" + visibleMagnitude).css("visibility","visible");
        $("#" + visibleMagnitude).css("display","block");
        console.log("Switching on magnitude view:" + visibleMagnitude);
    }