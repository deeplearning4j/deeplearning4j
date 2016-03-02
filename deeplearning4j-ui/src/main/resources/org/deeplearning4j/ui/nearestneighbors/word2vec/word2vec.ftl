<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <title>Nearest Neighbors</title>

    <!-- jQuery -->
    <script src="https://code.jquery.com/jquery-2.2.0.min.js"></script>

    <link href='http://fonts.googleapis.com/css?family=Roboto:400,300' rel='stylesheet' type='text/css'>

    <!-- Latest compiled and minified CSS -->
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css" integrity="sha384-1q8mTJOASx8j1Au+a5WDVnPi2lkFfwwEAa8hDDdjZlpLegxhjVME1fgjWPGmkzs7" crossorigin="anonymous" />

    <!-- Optional theme -->
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap-theme.min.css" integrity="sha384-fLW2N01lMqjakBkx3l/M9EahuwpSfeNvV63J5ezn3uZzapT0u7EYsXMjQV+0En5r" crossorigin="anonymous" />

    <!-- Latest compiled and minified JavaScript -->
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/js/bootstrap.min.js" integrity="sha384-0mSbJDEHialfmuBBQP6A4Qrprq5OVfW37PRR3j5ELqxss1yVqOtnepnHVP9aJ7xS" crossorigin="anonymous"></script>

    <!-- Booststrap Notify plugin-->
    <script src="/assets/bootstrap-notify.min.js"></script>
    <style>
        body {
        font-family: 'Roboto', sans-serif;
        color: #333;
        font-weight: 300;
        font-size: 16px;
        }
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
    </style>


    <link rel="stylesheet" href="/assets/css/simple-sidebar.css" />
    <link rel="stylesheet" href="/assets/css/style.css" />
    <script src="/assets/jquery-fileupload.js"></script>
    <script src="/assets/js/nearestneighbors/word2vec/app.js"></script>
</head>

<body>
<div style="position: fixed; top:0px; left: 0px; right: 0px; z-index: 1000;">
<table style="width: 100%; padding: 5px;" class="hd">
    <tbody>
    <tr>
        <td style="width: 48px;"><a href="/"><img src="/assets/deeplearning4j.img"  border="0"/></a></td>
        <td>DeepLearning4j UI</td>
        <td style="width: 128px;">&nbsp; <!-- placeholder for future use --></td>
    </tr>
    </tbody>
</table>
</div>
<br />
<br />
<div id="container" style="width: 100%;">
    <div id="wrapper" style="width: 100%;">

        <div id="sidebar-wrapper">

        </div>
        <div id="page-content-wrapper" style="text-align: center; width: 100%;">
            <div class="container-fluid" style="text-align: center; width: 100%;">
                <div id="instructions" style="text-align: left; display: inline-block;">
                <h2>k Nearest Neighbors</h2>
                <h4>
                    <ol>
                        <li>Upload a <b><i>vectorized</i></b> text file.</li>
                        <ul>
                            <li>The text file should be space-delimited.</li>
                            <li>Each row should be a feature vector separated by spaces.</li>
                            <li>If an individual feature has multiple words, use underscore to separate the words.</li>
                        </ul>
                        <li>Enter an integer value for k (number of nearest neighbors).</li>
                        <li>Then select a word on the left panel.</li>
                        <li>A list of k nearest neighbors will appear on this page.</li>
                        <li>Optional: Select a new word to update nearest neighbors.</li>
                    </ol>
                </h4>
                <br />
                </div>
                <div class="row" id="kform">
                    Number of nearest words to be returned:
                    <select name="k" id="k">
                        <option selected="selected">5</option>
                        <option>10</option>
                        <option>15</option>
                        <option>20</option>
                        <option>30</option>
                        <option>40</option>
                        <option>50</option>
                    </select>
                    <!-- <input type="text" name="k" id="k" value="5" /> -->
                </div>
                <div>
                    <div id="neighbors" style="text-align: left; width: 500px; display: inline-block;"></div>
                </div>

                <div style="text-align:center; width: 100%; position: fixed; bottom: 0px; left: 0px; margin-bottom: 15px;">
                     <div class="row" id="upload" style="display: inline-block; margin-right: 48px;">
                        <form encType="multipart/form-data" action="/word2vec/upload" method="POST" id="form">
                            <input name="file" type="file" style="width:300px; display: inline-block;" /><input type="submit" value="Upload WordVectors model" style="display: inline-block;"/>
                        </form>
                    </div>

                </div>
            </div>
        </div>
    </div>
</div>
</body>
</html>
