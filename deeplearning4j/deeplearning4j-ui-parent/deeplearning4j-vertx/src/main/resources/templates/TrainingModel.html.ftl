<!DOCTYPE html>

<!--~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ~
  ~
  ~ This program and the accompanying materials are made available under the
  ~ terms of the Apache License, Version 2.0 which is available at
  ~ https://www.apache.org/licenses/LICENSE-2.0.
  ~
  ~  See the NOTICE file distributed with this work for additional
  ~  information regarding copyright ownership.
  ~
  ~  Unless required by applicable law or agreed to in writing, software
  ~ distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
  ~ WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
  ~ License for the specific language governing permissions and limitations
  ~ under the License.
  ~
  ~ SPDX-License-Identifier: Apache-2.0
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~-->

<html lang="en">
    <head>

        <meta charset="utf-8">
        <title>${train\.pagetitle}</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">

        <link rel="stylesheet" href="/assets/webjars/coreui__coreui/2.1.9/dist/css/coreui.min.css">
        <link rel="stylesheet" href="/assets/css/style.css">


        <script src="/assets/webjars/jquery/3.4.1/dist/jquery.min.js"></script>
        <script src="/assets/webjars/popper.js/1.12.9/dist/umd/popper.min.js"></script>
        <script src="/assets/webjars/bootstrap/4.3.1/dist/js/bootstrap.min.js"></script>
        <script src="/assets/webjars/coreui__coreui/2.1.9/dist/js/coreui.min.js"></script>


        <!-- Icons -->
        <link rel="stylesheet" href="/assets/webjars/coreui__icons/0.3.0/css/coreui-icons.min.css"></script>


        <link rel="shortcut icon" href="/assets/img/favicon.ico">
    </head>

    <body class="app sidebar-show aside-menu-show">
        <header class="app-header navbar">
                <a class="header-text" href="#"><span>${train\.pagetitle}</span></a>
                <div id="sessionSelectDiv" style="display:none; float:right;">
                        <div style="color:white;">${train\.session\.label}</div>
                        <select id="sessionSelect" onchange='selectNewSession()'>
                        <option>(Session ID)</option>
                </select>
                </div>
                <div id="workerSelectDiv" style="display:none; float:right">
                        <div style="color:white;">${train\.session\.worker\.label}</div>
                        <select id="workerSelect" onchange='selectNewWorker()'>
                        <option>(Worker ID)</option>
                </select>
                </div>
        </header>
        <!-- End Header -->

        <div class="app-body">
            <div class="sidebar">
                <nav class="sidebar-nav">
                    <ul class="nav">
                        <li class="nav-item"><a class="nav-link" href="overview"><i class="nav-icon cui-chart"></i>${train\.nav\.overview}</a></li>
                        <li class="nav-item"><a class="nav-link" href="model"><i class="nav-icon cui-graph"></i>${train\.nav\.model}</a></li>
                        <li class="nav-item"><a class="nav-link" href="system"><i class="nav-icon cui-speedometer"></i>${train\.nav\.system}</a></li>
                        <li class="nav-item nav-dropdown">
                            <a class="nav-link nav-dropdown-toggle" href="#">
                                <i class="nav-icon cui-globe"></i> ${train\.nav\.language}
                            </a>
                            <ul class="nav-dropdown-items">
                                <li class="nav-item"><a class="nav-link" href="javascript:void(0);" onclick="languageSelect('en', 'overview')"><i class="icon-file-alt"></i>English</a></li>
                                <li class="nav-item"><a class="nav-link" href="javascript:void(0);" onclick="languageSelect('de', 'overview')"><i class="icon-file-alt"></i>Deutsch</a></li>
                                <li class="nav-item"><a class="nav-link" href="javascript:void(0);" onclick="languageSelect('ja', 'overview')"><i class="icon-file-alt"></i>日本語</a></li>
                                <li class="nav-item"><a class="nav-link" href="javascript:void(0);" onclick="languageSelect('zh', 'overview')"><i class="icon-file-alt"></i>中文</a></li>
                                <li class="nav-item"><a class="nav-link" href="javascript:void(0);" onclick="languageSelect('ko', 'overview')"><i class="icon-file-alt"></i>한글</a></li>
                                <li class="nav-item"><a class="nav-link" href="javascript:void(0);" onclick="languageSelect('ru', 'overview')"><i class="icon-file-alt"></i>русский</a></li>
                            </ul>
                        </li>
                    </ul>

                </nav>
            </div>

                <style>
                /* Graph */
                #layers {
                    height: 725px; /* IE8 */
                    height: 90vh;
                    width: 100%;
                    border: 2px solid #eee;
                }
                </style>

                    <!-- Start Content -->
                <div id="content">
                    <div class="row">
                        <div class="col">
                                <div id="layers"></div>
                        </div>

                <!-- Start Layer Details -->
                <div class="col" id="layerDetails" style="width:50pc">

                        <div class="box">
                                <div class="chart-header">
                                        <h2><b>${train\.model\.layerInfoTable\.title}</b></h2>
                                </div>
                                <div class="box-content">
                                        <table class="table table-bordered table-striped table-condensed" id="layerInfo"></table>
                                </div>
                        </div>

                        <div class="box">
                                <div class="chart-header">
                                        <h2><b>${train\.overview\.chart\.updateRatioTitle}</b></h2><p id="updateRatioTitleLog10"><b>: log<sub>10</sub></b></p>
                                        <ul class="nav tab-menu nav-tabs" style="position:absolute; margin-top: -60px; right: 27px;">
                                        <li id="mmRatioTab"><a href="javascript:void(0);" onclick="setSelectMeanMagChart('ratios')">${train\.model\.meanmag\.btn\.ratio}</a></li>
                                        <li id="mmParamTab"><a href="javascript:void(0);" onclick="setSelectMeanMagChart('paramMM')">${train\.model\.meanmag\.btn\.param}</a></li>
                                        <li id="mmUpdateTab"><a href="javascript:void(0);" onclick="setSelectMeanMagChart('updateMM')">${train\.model\.meanmag\.btn\.update}</a></li>
                                        </ul>
                                </div>
                                <div class="box-content">
                                        <div id="meanmag" class="center" style="height: 300px;" ></div>
                                        <p id="hoverdata"><span id="updateRatioTitleSmallLog10"><b>log<sub>
                                        10</sub> ${train\.overview\.chart\.updateRatioTitleShort}</b></span> <span id="yMeanMagnitudes">0</span>, <b>${train\.overview\.charts\.iteration}:</b> <span id="xMeanMagnitudes">0</span></p>
                                </div>
                        </div>
                                <div class="box">
                                        <div class="chart-header">
                                        <h2><b>${train\.model\.activationsChart\.title}</b></h2>
                                </div>
                                <div class="box-content">
                                        <div id="activations" class="center" style="height: 300px;" ></div>
                                        <p id="hoverdata"><b>${train\.model\.activationsChart\.titleShort}
                                :</b> <span id="yActivations">0</span>
                                        , <b>${train\.overview\.charts\.iteration}
                                :</b> <span id="xActivations">0</span></p>
                                </div>
                                </div>

                                <div class="box">
                                        <div class="chart-header">
                                        <h2><b>${train\.model\.paramHistChart\.title}</b></h2>
                                <div id="paramhistSelected" style="float: left"></div>
                                        <div id="paramHistButtonsDiv" style="float: right"></div>
                                        </div>
                                        <div class="box-content">
                                        <div id="parametershistogram" class="center" style="height: 300px;"></div>
                                        </div>
                                        </div>

                                        <div class="box">
                                        <div class="chart-header">
                                        <h2><b>${train\.model\.updateHistChart\.title}</b></h2>
                                <div id="updatehistSelected" style="float: left"></div>
                                        <div id="updateHistButtonsDiv" style="float: right"></div>
                                        </div>
                                        <div class="box-content">
                                        <div id="updateshistogram" class="center" style="height: 300px;"></div>
                                        </div>
                                        </div>

                                        <div class="box">
                                        <div class="chart-header">
                                        <h2><b>${train\.model\.lrChart\.title}</b></h2>
                                </div>
                                <div class="box-content">
                                        <div id="learningrate" class="center" style="height: 300px;" ></div>
                                        <p id="hoverdata"><b>${train\.model\.lrChart\.titleShort}
                                :</b> <span id="yLearningRate">0</span>
                                        , <b>${train\.overview\.charts\.iteration}
                                :</b> <span id="xLearningRate">0</span></p>
                                </div>
                                </div>

                                </div>
                                <!-- End Layer Details-->

                        <!-- Begin Zero State -->
                        <div class="col" id="zeroState">
                            <div class="box">
                                <div class="chart-header">
                                    <h2><b>Getting Started</b></h2>
                                </div>
                                <div class="box-content">
                                    <div class="page-header">
                                        <h1>Layer Visualization UI</h1>
                                    </div>
                                    <div class="row-fluid">
                                        <div class="span12">
                                            <h2>Overview</h2>
                                            <p>
                                                The layer visualization UI renders network structure dynamically. Users can inspect node layer parameters by clicking on the various elements of the GUI to see general information as well as overall network information such as performance.
                                            </p>
                                            <h2>Actions</h2>
                                            <p>On the <b>left</b>, you will find an interactive layer visualization.</p>
                                            <p>
                                        <ul>
                                            <li><b>Clicking</b> - Click on a layer to load network performance metrics.</li>
                                            <li><b>Scrolling</b>
                                                - Drag the GUI with your mouse or touchpad to move the model around. </li>
                                        </ul>
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                            <!-- End Zero State-->
                    </div>





                </div>
                    <!-- End Content -->
            </div> <!-- End Container -->
        </div> <!-- End Row Fluid-->

        <!-- Start JavaScript-->
        <script src="/assets/webjars/modernizr/2.8.3/modernizr.min.js"></script>
        <script src="/assets/webjars/flot/0.8.3/jquery.flot.js"></script>
        <script src="/assets/webjars/flot/0.8.3/jquery.flot.pie.js"></script>
        <script src="/assets/webjars/flot/0.8.3/jquery.flot.stack.js"></script>
        <script src="/assets/webjars/flot/0.8.3/jquery.flot.resize.min.js"></script>
        <script src="/assets/webjars/chosen/0.9.8/chosen/chosen.jquery.min.js"></script>
        <script src="/assets/webjars/uniform/2.1.2/jquery.uniform.min.js"></script>
        <script src="/assets/webjars/noty/2.2.2/jquery.noty.packaged.js"></script>
        <script src="/assets/webjars/jquery-raty/2.5.2/jquery.raty.min.js"></script>
        <script src="/assets/webjars/imagesloaded/2.1.1/jquery.imagesloaded.min.js"></script>
        <script src="/assets/webjars/masonry/3.1.5/masonry.pkgd.min.js"></script>
        <script src="/assets/webjars/jquery-knob/1.2.2/jquery.knob.min.js"></script>
        <script src="/assets/webjars/jquery.sparkline/2.1.2/jquery.sparkline.min.js"></script>
        <script src="/assets/webjars/retinajs/0.0.2/retina.js"></script>
        <script src="/assets/webjars/dagre/0.8.4/dist/dagre.min.js"></script>
        <script src="/assets/webjars/cytoscape/3.3.3/dist/cytoscape.min.js"></script>
        <script src="/assets/webjars/cytoscape-dagre/2.1.0/cytoscape-dagre.js"></script>
        <script src="/assets/webjars/github-com-jboesch-Gritter/1.7.4/jquery.gritter.js"></script>

        <script src="/assets/js/train/model.js"></script> <!-- Charts and tables are generated here! -->
        <script src="/assets/js/train/model-graph.js"></script> <!-- Layer graph generated here! -->
        <script src="/assets/js/train/train.js"></script> <!-- Common (lang selection, etc) -->
        <script src="/assets/js/counter.js"></script>



            <!-- Execute once on page load -->
       <script>
               $(document).ready(function () {
                   renderModelGraph();
                   renderModelPage(true);
               });
       </script>

               <!-- Execute periodically (every 2 sec) -->
        <script>
                setInterval(function () {
                    renderModelPage(false);
                }, 2000);
        </script>
    </body>
</html>
