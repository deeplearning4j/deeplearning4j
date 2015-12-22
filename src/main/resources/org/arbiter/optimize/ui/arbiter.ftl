<!DOCTYPE html>
<html>
<head>
    <style type="text/css">
        /* Color and style reference.
            To change: do find + replace on comment + color

            heading background:         headingbgcol        #063E53            //Old candidates: #3B5998
            heading text color:         headingtextcolor    white

        */

        html, body {
            width: 100%;
            height: 100%;
            padding-top: 20px;
            padding-left: 20px;
            padding-right: 20px;
            padding-bottom: 20px;
        }

        .bgcolor {
            background-color: #D1C5B7; /* OLD: #D2E4EF;*/
        }

        h1 {
            font-family: Georgia, Times, 'Times New Roman', serif;
            font-size: 28px;
            font-style: bold;
            font-variant: normal;
            font-weight: 500;
            line-height: 26.4px;
        }

        h3 {
            font-family: Georgia, Times, 'Times New Roman', serif;
            font-size: 16px;
            font-style: normal;
            font-variant: normal;
            font-weight: 500;
            line-height: 26.4px;
        }

        table.resultsTable {
            border-collapse:collapse;
            background-color: white;
            /*border-collapse: collapse;*/
            border-radius: 5px;
            -moz-border-radius:5px;
        }

        table.resultsTable td, table.resultsTable tr, table.resultsTable th {
            border:solid black 1px;
        }

        table.resultsTable th {
            background-color: #063E53;
            color: white;
            padding-left: 4px;
            padding-right: 4px;
        }
        table.resultsTable td {
            background-color: white;
            padding-left: 4px;
            padding-right: 4px;
        }

        div.outerelements {
            /*
            padding-top: 20px;
            padding-left: 20px;
            padding-right: 20px;
            padding-bottom: 20px;
            */
        }

        #accordion, #accordion2 {
            padding-bottom: 20px;
        }

        #accordion .ui-accordion-header, #accordion2 .ui-accordion-header {
            background-color: /*headingbgcolor*/#063E53;      /*Color when collapsed*/
            color: /*headingtextcolor*/white;
            font-family: Georgia, Times, 'Times New Roman', serif;
            font-size: 20px;
            font-style: bold;
            font-variant: normal;
            margin: 0px;
            background-image: none;     /* Necessary, otherwise color changes don't make a difference */
        }

        /*
        #accordion .ui-accordion-header.ui-state-active {
            background-color: pink;
            background-image: none;
        }*/

        #accordion .ui-accordion-content {
            width: 100%;
            background-color: white;    /*background color of accordian content (elements in front may have different color */
            color: black;  /* text etc color */
            font-size: 10pt;
            line-height: 16pt;
        }


    </style>
    <title>Arbiter UI</title>
</head>
<body class="bgcolor">

<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="http://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/css/bootstrap.min.css">

<script src="//ajax.googleapis.com/ajax/libs/jquery/1/jquery.min.js"></script>
<link rel="stylesheet" href="//code.jquery.com/ui/1.11.4/themes/smoothness/jquery-ui.css">
<script src="//code.jquery.com/jquery-1.10.2.js"></script>
<script src="//code.jquery.com/ui/1.11.4/jquery-ui.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js"></script>
<script src="http://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/js/bootstrap.min.js"></script>
<script type="text/javascript" src="http://localhost:8080/assets/jquery.tablesorter.min.js"></script>

<script>
    //Store last update times:
    var lastStatusUpdateTime = -1;
    var lastSettingsUpdateTime = -1;
    var lastResultsUpdateTime = -1;


    //Set basic interval function to do updates
    setInterval(function(){
        //Get the update status, and do something with it:
        $.get("/lastUpdate",function(data){
            //Encoding: matches names in UpdateStatus class
            var jsonObj = JSON.parse(JSON.stringify(data));
            var statusTime = jsonObj['statusUpdateTime'];
            var settingsTime = jsonObj['settingsUpdateTime'];
            var resultsTime = jsonObj['resultsUpdateTime'];
            console.log("Last update times: " + statusTime + ", " + settingsTime + ", " + resultsTime);

            //Check last update times for each part of document, and update as necessary
            //First section: summary status
            if(lastStatusUpdateTime != statusTime){
                //Get JSON: address set by SummaryStatusResource
                $.get("/summary",function(data){
                    var jsonObj = JSON.parse(JSON.stringify(data));

                    var summaryStatusDiv = $('#statusdiv');
                    var components = jsonObj['renderableComponents'];
                    if(!components) summaryStatusDiv.html('');
                    summaryStatusDiv.html('');

                    var len = (!components ? 0 : components.length);
                    for(var i=0; i<len; i++){
                        var c = components[i];
                        var temp = getComponentHTML(c);
                        summaryStatusDiv.append(temp);
                    }
                });

                lastStatusUpdateTime = statusTime;
            }

            //Second section: Optimization settings
            if(lastSettingsUpdateTime != settingsTime){
                //Get JSON: address set by ConfigResource
                $.get("/config",function(data){
                    var jsonObj = JSON.parse(JSON.stringify(data));
//                    console.log("summary status: " + jsonObj);
                    console.log("Config JSON keys: " + Object.keys(jsonObj));

                    var components = jsonObj['renderableComponents'];

                    var configDiv = $('#settingsdiv');
                    configDiv.html('');

                    var len = (!components ? 0 : components.length);
                    for(var i=0; i<len; i++){
                        var c = components[i];
                        var temp = getComponentHTML(c);
                        configDiv.append(temp);
                    }
                });

                lastSettingsUpdateTime = settingsTime;
            }

            //Third section: Summary results table (summary info for each candidate)
            if(lastResultsUpdateTime != resultsTime){

                //Get JSON; address set by ResultsResource
                $.get("/results",function(data){
                    //Expect an array of CandidateStatus type objects here
                    drawResultTable(data);
                });

                lastResultsUpdateTime = resultsTime;
            }
        })
    },4000);

    function getComponentHTML(renderableComponent){
        var key = Object.keys(renderableComponent)[0];
        var type = renderableComponent[key]['componentType'];

        switch(type){
            case "string":
                var s = renderableComponent[key]['string'];
                return s.replace(new RegExp("\n",'g'),"<br>");
            case "simpletable":
                return createTable(renderableComponent[key],"someidhere");
            default:
                return "UNKNOWN OBJECT";
        }

    }

    function createTable(tableObj,tableId){
        //Expect RenderComponentTable
        var header = tableObj['header'];
        var values = tableObj['table'];
        var nRows = (values ? values.length : 0);

        var table = $("<table id=\"" + tableId + "\">");
        if(header){
            var headerRow = $("<tr>");
            var len = header.length;
            for( var i=0; i<len; i++ ){
                headerRow.append($("<th>" + header[i] + "</th>"));
            }
            headerRow.append($("</tr>"));
            table.append(headerRow);
        }

        if(values){
            for( var i=0; i<nRows; i++ ){
                var row = $("<tr>");
                var rowValues = values[i];
                var len = rowValues.length;
                for( var j=0; j<len; j++ ){
                    row.append($("<td>"+rowValues[j]+"</td>"));
                }
                row.append($("</tr>"));
                table.append(row);
            }
        }

        table.append($("</table>"));
        return table;
    }

    function drawResultTable(resultArray){
//        var resultsTable = $('#resultsTable');
//        /*$('#resultsTable tr').not(function(){if ($(this).has('th').length){return true}}).remove(); //Remove all rows, but keep header*/
//        $('#resultstablebody').empty();
//
////        resultsTable.append($("<tbody>"));
//        var len = (!resultArray ? 0 : resultArray.length);
//        for(var i=0; i<len; i++){
//            var row = $("<tr />");
//            row.append($("<td>" + resultArray[i].index + "</td>"));
//            row.append($("<td>" + resultArray[i].score + "</td>"));
//            row.append($("<td>" + resultArray[i].status + "</td>"));
//            resultsTable.append(row);
//        }
////        resultsTable.append($("</tbody>"));

        var tableBody = $('#resultstablebody');
        tableBody.empty();

        var len = (!resultArray ? 0 : resultArray.length);
        for(var i=0; i<len; i++){
            var row = $("<tr />");
            row.append($("<td>" + resultArray[i].index + "</td>"));
            row.append($("<td>" + resultArray[i].score + "</td>"));
            row.append($("<td>" + resultArray[i].status + "</td>"));
            tableBody.append(row);
        }

        $('.tablesorter').trigger('update');
    }

</script>
<script>
    $(function() {
        $( "#accordion" ).accordion({
            collapsible: true,
            heightStyle: "content"
        });
    });
    $(function() {
        $( "#accordion2" ).accordion({
            collapsible: true,
            heightStyle: "content"
        });
    });
</script>

<script type="text/javascript">
    $(function(){
        $('#resultsTable').tablesorter();
    });
</script>

<div class="outerelements" id="heading">
    <h1>Arbiter</h1>
</div>


<div class="outerelements" id="status">
    <div id="accordion" class="hcol2">
        <h3 class="hcol2 headingcolor ui-accordion-header">Summary</h3>
        <div class="statusdiv" id="statusdiv">
            Table with: best score, best index, total runtime, date, etc
        </div>
    </div>
</div>

<div class="outerelements" id="settings">
    <div id="accordion2">
        <h3 class="ui-accordion-header headingcolor">Optimization Settings</h3>
        <div class="settingsdiv" id="settingsdiv">
            Collapseable box with settings for hyperparameter space settings etc
        </div>
    </div>
</div>


<div class="outerelements" id="results">
    <h3>Results</h3>
    <div class="resultsdiv" id="resultsdiv">
        <table style="width:100%" id="resultsTable" class="resultsTable tablesorter">
            <thead><tr> <th>ID</th> <th>Score</th> <th>Status</th> </tr></thead>
            <tbody id="resultstablebody"></tbody>
        </table>

    </div>
    Collapsable table goes here. Summary results when collapsed, full results when expanded.
    Also sortable by ID, status, score, runtime etc.
</div>


</body>
</html>