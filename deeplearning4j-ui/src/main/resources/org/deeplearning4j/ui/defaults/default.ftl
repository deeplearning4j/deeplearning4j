<!DOCTYPE html>
<html>
    <head>
        <title>DeepLearning4j UI</title>
        <meta charset="utf-8" />

        <link href='http://fonts.googleapis.com/css?family=Roboto:400,300' rel='stylesheet' type='text/css'>

        <!-- Latest compiled and minified CSS -->
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css" integrity="sha384-1q8mTJOASx8j1Au+a5WDVnPi2lkFfwwEAa8hDDdjZlpLegxhjVME1fgjWPGmkzs7" crossorigin="anonymous" />

        <!-- Optional theme -->
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap-theme.min.css" integrity="sha384-fLW2N01lMqjakBkx3l/M9EahuwpSfeNvV63J5ezn3uZzapT0u7EYsXMjQV+0En5r" crossorigin="anonymous" />

        <!-- Latest compiled and minified JavaScript -->
        <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/js/bootstrap.min.js" integrity="sha384-0mSbJDEHialfmuBBQP6A4Qrprq5OVfW37PRR3j5ELqxss1yVqOtnepnHVP9aJ7xS" crossorigin="anonymous"></script>

        <!-- jQuery -->
        <script src="https://code.jquery.com/jquery-2.2.0.min.js"></script>
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

                    margin-right: 64px;
            }
            .hd-small {
                    background-color: #000000;
                    font-size: 14px;
                    color: #FFFFFF;
                }
        </style>
    </head>
    <body>
    <table style="width: 100%; padding: 5px;" class="hd">
        <tbody>
            <tr>
                <td style="width: 48px;"><img src="/assets/deeplearning4j.img"  border="0"/></td>
                <td>DeepLearning4j UI</td>
                <td style="width: 128px;">&nbsp; <!-- placeholder for future use --></td>
            </tr>
        </tbody>
    </table>

    <br />
    <br />
    <br />
<!--
    Here we should provide nav to available modules:
    T-SNE visualization
    NN activations
    HistogramListener renderer
 -->
<div style="width: 100%; text-align: center">
    <div class="block">
        <!-- TSNE block -->
        <b>T-SNE</b><br/><br/>
        <a href="/tsne"><img src="/assets/i_plot.img" border="0" /></a><br/><br/>
        <div style="text-align: left; margin: 5px;">
            &nbsp;Plot T-SNE data uploaded by user or retrieved from DL4j.
        </div>
    </div>

    <div class="block">
        <!-- W2V block -->
        <b>WordVectors</b><br/><br/>
        <a href="/word2vec"><img src="/assets/i_nearest.img" border="0" /></a><br/><br/>
        <div style="text-align: left; margin: 5px;">
            &nbsp;wordsNearest UI for WordVectors (GloVe/Word2Vec compatible)
        </div>
    </div>
<!--
    <div class="block">
         Activations block
        <b>Activations</b><br/><br/>
        <a href="/activations"><img src="/assets/i_ladder.img" border="0" /></a><br/><br/>
        <div style="text-align: left; margin: 5px;">
            &nbsp;Neural network activations retrieved from DL4j.
        </div>
    </div>
    -->
    <div class="block">
        <!-- Histogram block -->
        <b>Histo</b><br/><br/>
        <a href="/weights"><img src="/assets/i_histo.img" border="0" /></a><br/><br/>
        <div style="text-align: left; margin: 5px;">
            &nbsp;Neural network scores retrieved from DL4j during training.
        </div>
    </div>
</div>

    </body>
</html>