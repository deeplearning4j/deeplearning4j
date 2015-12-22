<style>
    body {background-color:#EAF0F4;}

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

    div.outerelements {
        padding-top: 20px;
        padding-left: 20px;
        padding-right: 20px;
        padding-bottom: 20px;
    }

    #accordion {
        margin: 8px auto;
        font-family: Georgia, Times, 'Times New Roman', serif;
        font-size: 16px;
    }
    #accordion .ui-accordion-header {
        font-family: Georgia, Times, 'Times New Roman', serif;
        font-size: 16pt;
        background-color: #ccc;
        margin: 0px;
    }
    #accordion .ui-accordion-header a {
        color: #fff;
        line-height: 42px;
        display: block;
        font-size: 16pt;
        width: 100%;
        text-indent: 10px;
        text-shadow: 1px 1px 0px rgba(0,0,0,0.2);
        border-right: 1px solid rgba(0, 0, 0, .2);
        border-left: 1px solid rgba(0, 0, 0, .2);
        border-bottom: 1px solid rgba(0, 0, 0, .2);
        border-top: 1px solid rgba(250, 250, 250, .2);
    }
    #accordion .ui-accordion-content {
        width: 100%;
        background-color: #f3f3f3;
        color: #777;
        font-size: 10pt;
        line-height: 16pt;
        box-shadow: inset 0px 1px 1px 0px rgba(0, 0, 0, .2),
        inset 0px -1px 0px 0px rgba(0, 0, 0, .4);
    }
    #accordion .ui-accordion-content > * {
        margin: 0;
        padding: 10px;
    }
    #accordion .ui-accordion-content a {
        color: #777;
    }

    #accordion2 {
        margin: 8px auto;
        font-family: Georgia, Times, 'Times New Roman', serif;
        font-size: 16px;
    }
    #accordion2 .ui-accordion-header {
        font-family: Georgia, Times, 'Times New Roman', serif;
        font-size: 16pt;
        background-color: #ccc;
        margin: 0px;
    }
    #accordion2 .ui-accordion-header a {
        color: #fff;
        line-height: 42px;
        display: block;
        font-size: 16pt;
        width: 100%;
        text-indent: 10px;
        text-shadow: 1px 1px 0px rgba(0,0,0,0.2);
        border-right: 1px solid rgba(0, 0, 0, .2);
        border-left: 1px solid rgba(0, 0, 0, .2);
        border-bottom: 1px solid rgba(0, 0, 0, .2);
        border-top: 1px solid rgba(250, 250, 250, .2);
    }
    #accordion2 .ui-accordion-content {
        width: 100%;
        background-color: #f3f3f3;
        color: #777;
        font-size: 10pt;
        line-height: 16pt;
        box-shadow: inset 0px 1px 1px 0px rgba(0, 0, 0, .2),
        inset 0px -1px 0px 0px rgba(0, 0, 0, .4);
    }
    #accordion2 .ui-accordion-content > * {
        margin: 0;
        padding: 10px;
    }
    #accordion2 .ui-accordion-content a {
        color: #777;
    }
</style>

<html>
<head>
    <title>Arbiter UI</title>
</head>
<body>

<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="http://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/css/bootstrap.min.css">

<script src="//ajax.googleapis.com/ajax/libs/jquery/1/jquery.min.js"></script>
<link rel="stylesheet" href="//code.jquery.com/ui/1.11.4/themes/smoothness/jquery-ui.css">
<script src="//code.jquery.com/jquery-1.10.2.js"></script>
<script src="//code.jquery.com/ui/1.11.4/jquery-ui.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js"></script>
<script src="http://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/js/bootstrap.min.js"></script>
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
//            console.log(data);
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
//                    console.log("summary status: " + jsonObj);
                    console.log("Summary status JSON keys: " + Object.keys(jsonObj));

                    var summaryStatusDiv = $('#statusdiv');
                    var components = jsonObj['renderableComponents'];
                    if(!components) summaryStatusDiv.html('');
                    //console.log("Number of renderable components: " + components.length);
                    //TODO: put each into a div or something... (just using last here)
                    var temp;
                    var len = (!components ? 0 : components.length);
                    for(var i=0; i<len; i++){
                        var c = components[i];
//                        console.log("Component " + i + " keys: " + Object.keys(c));
                        temp = getComponentHTML(c);
                    }

                    summaryStatusDiv.html('');
//                    summaryStatusDiv.html("Data: " + jsonObj.toString());
                    summaryStatusDiv.html(temp);

                    //Parse
                    //Update elements
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
                    if(!components) configDiv.html('');

                    //TODO: put each into a div or something... (just using last here)
                    var temp;
                    var len = (!components ? 0 : components.length);
                    for(var i=0; i<len; i++){
                        var c = components[i];
//                        console.log("Component " + i + " keys: " + Object.keys(c));
                        temp = getComponentHTML(c);
                    }

                    configDiv.html(temp);

                });

                lastSettingsUpdateTime = settingsTime;
            }

            //Third section: Summary results table (summary info for each candidate)
            if(lastResultsUpdateTime != resultsTime){

                //Get JSON; address set by ResultsResource
                $.get("/results",function(data){
                    //Expect an array of CandidateStatus type objects here
//                    var jsonObj = JSON.parse(JSON.stringify(data));

                    drawResultTable(data);

//                    var toPost = "";
//                    var len = (!data ? 0 : data.length);
//                    for( var i=0; i<len; i++ ){
//                        toPost = toPost.concat("<br>",data[i].index," - ",data[i].score," - ",data[i].status);
////                        toPost = toPost.concat("<br>",jsonObj[i].index," - ",jsonObj[i].score," - ",jsonObj[i].status);
//                    }
//
////                    var jsonObj = JSON.parse(JSON.stringify(data));
//
//                    //For now: just post JSON stuff as string (to test)
//                    var resultsDiv = $('#resultsdiv');
////                    resultsDiv.html(jsonObj[0]);
//                    resultsDiv.html(toPost);
//                    console.log("Results: " + jsonObj);
                });

                lastResultsUpdateTime = resultsTime;
            }
        })
    },4000);

    function getComponentHTML(renderableComponent){
//        var type = renderableComponent['componentType'];
        var key = Object.keys(renderableComponent)[0];
        var type = renderableComponent[key]['componentType'];

        switch(type){
            case "string":
                var s = renderableComponent[key]['string'];
                return s.replace(new RegExp("\n",'g'),"<br>");
            default:
                return "UNKNOWN OBJECT";
        }

    }

    function drawResultTable(resultArray){
        var resultsTable = $('#resultsTable');

//        $('#resultsTable tbody tr').html('');   //Remove all rows, but keep header
//        $("#resultsTable > tbody > tr").remove();
        $('#resultsTable tr').not(function(){if ($(this).has('th').length){return true}}).remove();

        var len = (!resultArray ? 0 : resultArray.length);
        for(var i=0; i<len; i++){
            var row = $("<tr />");
//            row.append("<td>" + resultArray[i].index + "</td>");
//            row.append("<td>" + resultArray[i].score + "</td>");
//            row.append("<td>" + resultArray[i].status + "</td>");
//            row.append("</tr>");
            row.append($("<td>" + resultArray[i].index + "</td>"));
            row.append($("<td>" + resultArray[i].score + "</td>"));
            row.append($("<td>" + resultArray[i].status + "</td>"));
            resultsTable.append(row);
        }
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

<div class="outerelements" id="heading">
    <h1>Arbiter</h1>
</div>


<div class="outerelements" id="status">
    <div id="accordion">
        <h3 class="ui-accordion-header">Summary</h3>
        <div class="statusdiv" id="statusdiv">
            Table with: best score, best index, total runtime, date, etc
        </div>
    </div>
</div>

<div class="outerelements" id="settings">
    <div id="accordion2">
        <h3 class="ui-accordion-header">Optimization Settings</h3>
        <div class="settingsdiv" id="settingsdiv">
            Collapseable box with settings for hyperparameter space settings etc
        </div>
    </div>
</div>


<div class="outerelements" id="results">
    <h3>Results</h3>
    <div class="resultsdiv" id="resultsdiv">
        <table style="width:100%" id="resultsTable">
            <tr> <th>ID</th> <th>Score</th> <th>Status</th> </tr>

        </table>

    </div>
    Collapsable table goes here. Summary results when collapsed, full results when expanded.
    Also sortable by ID, status, score, runtime etc.
</div>


</body>
</html>