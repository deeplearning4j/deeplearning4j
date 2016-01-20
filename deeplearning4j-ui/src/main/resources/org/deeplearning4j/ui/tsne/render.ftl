<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Tsne renders</title>

    <!-- jQuery -->
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



    <style>
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
                $('#form').hide();

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
<div id="embed"></div>

<h4>Tsne embeddings</h4>

<h4 class="hero">Upload a file</h4>
<div class="row" id="upload">
    <form encType="multipart/form-data" action="/api/upload" method="POST" id="form">
        <input name="file" type="file">
        <br>
        <input type="submit">
    </form>
</div>

<h4>If a file is already present on the server, specify the name.</h4>
<div class="row" id="filebutton">
    <input type="text" id="filename"/>
    <button id="filenamebutton">Submit</button>
</div>

</body>

</html>