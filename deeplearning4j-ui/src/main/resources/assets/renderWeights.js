

    function appendHistogram(values,selector) {
        // A formatter for counts.
        var formatCount = d3.format(",.0f");

        var margin = {top: 10, right: 30, bottom: 30, left: 30},
                width = 650 - margin.left - margin.right,
                height = 400 - margin.top - margin.bottom;
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
        var margin = {top: 30, right: 20, bottom: 30, left: 50},
                width = 650 - margin.left - margin.right,
                height = 350 - margin.top - margin.bottom;

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

        // Set the dimensions of the canvas / graph
        var margin = {top: 30, right: 20, bottom: 60, left: 50},
                width = 650 - margin.left - margin.right,
                height = 350 - margin.top - margin.bottom;

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


setInterval(function() {
        $.get( "${path}" + "/updated", function( data ) {
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

                if(!model || !gradient || !score || !scores || !updateMagnitudes || !paramMagnitudes )
                    return;
                $('.score').html('' + score);


                $('#scores .chart').html('');
                var scdiv = '<div class="scorechart"></div>';
                $('#scores .chart').append(scdiv);
                appendLineChart(scores,'#scores .chart');
/*
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
                    var div = '<div class="' + key + '"><h4>' + key + '</h4></div>';
                    $(selectorModel).append(div);
                    $(selectorGradient).append(div);
                    appendHistogram(model[key]['dataBuffer'],selectorModel + ' .' + key);
                    appendHistogram(gradient[key]['dataBuffer'],selectorGradient + ' .' + key);
                }
                */
/*
                //Plot mean magnitudes: weights and params
                $('#magnitudes .charts').html('');
                for(var i=0; i<updateMagnitudes.length; i++ ){
                    //Maps:
                    var mapParams = paramMagnitudes[i];
                    var mapUpdates = updateMagnitudes[i];

                    var selectorModel = '#magnitudes .charts'
                    var div = '<div class="layer' + i + 'param"><h4>Layer ' + i + ' Parameter Mean Magnitudes</h4></div>';
                    $(selectorModel).append(div);
                    appendMultiLineChart(mapParams,selectorModel + ' .layer' + i + 'param');
                    div = '<div class="layer' + i + 'grad"><h4>Layer ' + i + ' Update/Gradient Mean Magnitudes</h4></div>';
                    $(selectorModel).append(div);
                    appendMultiLineChart(mapUpdates,selectorModel + ' .layer' + i + 'grad');
                }
*/
                var time = new Date(updateTime);
                $('#updatetime').html(time.customFormat("#DD#/#MM#/#YYYY# #hhh#:#mm#:#ss#"));
            });
        });

    },1000);