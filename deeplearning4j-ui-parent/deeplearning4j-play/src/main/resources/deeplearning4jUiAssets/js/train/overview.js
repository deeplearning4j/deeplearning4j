



/* ---------- Score vs. Iteration Chart ---------- */
function renderScoreChart() {

    if($("#scoreiterchart").length)
    {
        var sin = [];

        for (var i = 0; i < 14; i += 0.5) {
            sin.push([i, Math.sin(i)/i]);
        }

        var plot = $.plot($("#scoreiterchart"),
            [ { data: sin, label: "sin(x)/x"} ], {
                series: {
                    lines: { show: true,
                        lineWidth: 2,
                    },
                    points: { show: true },
                    shadowSize: 2
                },
                grid: { hoverable: true,
                    clickable: true,
                    tickColor: "#dddddd",
                    borderWidth: 0
                },
                yaxis: { min: -1.2, max: 1.2 },
                colors: ["#FA5833", "#2FABE9"]
            });

        function showTooltip(x, y, contents) {
            $('<div id="tooltip">' + contents + '</div>').css( {
                position: 'absolute',
                display: 'none',
                top: y + 5,
                left: x + 5,
                border: '1px solid #fdd',
                padding: '2px',
                'background-color': '#dfeffc',
                opacity: 0.80
            }).appendTo("body").fadeIn(200);
        }

        var previousPoint = null;
        $("#scoreiterchart").bind("plothover", function (event, pos, item) {
            $("#x").text(pos.x.toFixed(2));
            $("#y").text(pos.y.toFixed(2));

            if (item) {
                if (previousPoint != item.dataIndex) {
                    previousPoint = item.dataIndex;

                    $("#tooltip").remove();
                    var x = item.datapoint[0].toFixed(2),
                        y = item.datapoint[1].toFixed(2);

                    showTooltip(item.pageX, item.pageY,
                        item.series.label + " of " + x + " = " + y);
                }
            }
            else {
                $("#tooltip").remove();
                previousPoint = null;
            }
        });



        // $("#scoreiterchart").bind("plotclick", function (event, pos, item) {
        //     if (item) {
        //         $("#clickdata").text("You clicked point " + item.dataIndex + " in " + item.series.label + ".");
        //         plot.highlight(item.series, item.datapoint);
        //     }
        // });
    }

}