<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <title>T-SNE renders</title>

    <!-- jQuery -->
    <!--
    <script src="//ajax.googleapis.com/ajax/libs/jquery/1/jquery.min.js"></script>
    <script src="/assets/d3.min.js"></script>
    <script src="/assets/render.js"></script>
    <link href='http://fonts.googleapis.com/css?family=Roboto:400,300' rel='stylesheet' type='text/css'>
    <link rel="stylesheet" href="/assets/bootstrap-3.3.4-dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="/assets/bootstrap-3.3.4-dist/css/bootstrap-theme.min.css">
    <link rel="stylesheet" href="/assets/css/simple-sidebar.css">
    <link rel="stylesheet" href="/assets/css/style.css">
    <script src="https://code.jquery.com/jquery-2.1.3.min.js"></script>
    <script src="/assets/jquery-fileupload.js"></script>
    <script src="/assets/bootstrap-3.3.4-dist/js/bootstrap.min.js"></script>
    -->
    <!-- jQuery -->
    <script src="https://code.jquery.com/jquery-2.2.0.min.js"></script>

    <link href='http://fonts.googleapis.com/css?family=Roboto:400,300' rel='stylesheet' type='text/css'>

    <!-- Latest compiled and minified CSS -->
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css" integrity="sha384-1q8mTJOASx8j1Au+a5WDVnPi2lkFfwwEAa8hDDdjZlpLegxhjVME1fgjWPGmkzs7" crossorigin="anonymous" />

    <!-- Optional theme -->
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap-theme.min.css" integrity="sha384-fLW2N01lMqjakBkx3l/M9EahuwpSfeNvV63J5ezn3uZzapT0u7EYsXMjQV+0En5r" crossorigin="anonymous" />

    <!-- Latest compiled and minified JavaScript -->
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/js/bootstrap.min.js" integrity="sha384-0mSbJDEHialfmuBBQP6A4Qrprq5OVfW37PRR3j5ELqxss1yVqOtnepnHVP9aJ7xS" crossorigin="anonymous"></script>


    <!-- d3 -->
    <script src="//d3js.org/d3.v3.min.js" charset="utf-8"></script>

    <!-- dl4j plot setup -->
    <script src="/assets/renderTsne.js"></script>

    <script src="/assets/jquery-fileupload.js"></script>
    <style>
        .hd {
        background-color: #000000;
        font-size: 18px;
        color: #FFFFFF;
        }
        .block {
        width: 250px;
        height: 350px;
        display: inline-block;
        border: 1px solid #DEDEDE;
        margin-right: 64px;
        }
        .hd-small {
        background-color: #000000;
        font-size: 14px;
        color: #FFFFFF;
        }

        body {
            font-family: 'Roboto', sans-serif;
            color: #333;
            font-weight: 300;
            font-size: 16px;
        }
        svg {
            border: 1px solid #333;
        }
        #wrap {
            width: 800px;
            margin-left: auto;
            margin-right: auto;
        }
        #embed {
            margin-top: 10px;
        }
        h1 {
            text-align: center;
            font-weight: normal;
        }
        .tt {
            margin-top: 10px;
            background-color: #EEE;
            border-bottom: 1px solid #333;
            padding: 5px;
        }
        .txth {
            color: #F55;
        }
        .cit {
            font-family: courier;
            padding-left: 20px;
            font-size: 14px;
        }
    </style>

    <script>
        $(document).ready(function() {
            $('#filenamebutton').click(function() {
                document.getElementById('form').reset();
                $('#form').hide();
                var filename = $('#filename').val();
                $('#filename').val('');
                updateFileName(filename);
                drawTsne();
            });

            $('#form').fileUpload({success : function(data, textStatus, jqXHR){
                var fullPath = document.getElementById('form').value;
                var filename = data['name'];
                if (fullPath) {
                    var startIndex = (fullPath.indexOf('\\') >= 0 ? fullPath.lastIndexOf('\\') : fullPath.lastIndexOf('/'));
                    var filename = fullPath.substring(startIndex);
                    if (filename.indexOf('\\') === 0 || filename.indexOf('/') === 0) {
                        filename = filename.substring(1);
                    }
                }

                document.getElementById('form').reset();
                //$('#form').hide();

                updateFileName(filename);
                drawTsne();

            },error : function(err) {
                console.log(err);
            }});

            function updateFileName(name) {
                $.ajax({
                    url: '/api/update',
                    type: 'POST',
                    dataType: 'json',
                    data: JSON.stringify({"url" : name}),
                    cache: false,
                    success: function(data, textStatus, jqXHR) {


                    },
                    error: function(jqXHR, textStatus, errorThrown) {
                        // Handle errors here
                        console.log('ERRORS: ' + textStatus);
                    },
                    complete: function() {
                    }
                });
            }

        }) ;

    </script>

</head>

<body>
<table style="width: 100%; padding: 5px;" class="hd">
    <tbody>
    <tr>
        <td style="width: 48px;"><a href="/"><img src="/assets/deeplearning4j.img"  border="0"/></a></td>
        <td>DeepLearning4j UI</td>
        <td style="width: 128px;">&nbsp; <!-- placeholder for future use --></td>
    </tr>
    </tbody>
</table>

<br />
<div style="text-align: center">
    <div id="embed" style="display: inline-block; width: 1024px; height: 700px; border: 1px solid #DEDEDE;"></div>
</div>
<br/>
<br/>
<div style="text-align:center; width: 100%; position: fixed; bottom: 0px; left: 0px; margin-bottom: 15px;">
    <div style="display: inline-block; margin-right: 48px;">
        <h5>Upload a file to UI server.</h5>
        <form encType="multipart/form-data" action="/api/upload" method="POST" id="form">
        <div>

            <input name="file" type="file" style="width:300px; display: inline-block;" /><input type="submit" value="Upload file" style="display: inline-block;"/>

        </div>
        </form>
    </div>

    <div style="display: inline-block;">
        <h5>If a file is already present on the server, specify the path/name.</h5>
        <div id="filebutton">
            <input type="text" id="filename"/>
            <button id="filenamebutton">Submit</button>
        </div>
    </div>
</div>
</body>

</html>