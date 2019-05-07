
package org.deeplearning4j.ui.views.html.training

import play.twirl.api._
import play.twirl.api.TemplateMagic._


     object TrainingModel_Scope0 {
import models._
import controllers._
import play.api.i18n._
import views.html._
import play.api.templates.PlayMagic._
import play.api.mvc._
import play.api.data._

class TrainingModel extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template1[org.deeplearning4j.ui.api.I18N,play.twirl.api.HtmlFormat.Appendable] {

  /**/
  def apply/*1.2*/(i18n: org.deeplearning4j.ui.api.I18N):play.twirl.api.HtmlFormat.Appendable = {
    _display_ {
      {


Seq[Any](format.raw/*1.40*/("""
"""),format.raw/*2.1*/("""<!DOCTYPE html>

<!--~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ~ Copyright (c) 2015-2018 Skymind, Inc.
  ~
  ~ This program and the accompanying materials are made available under the
  ~ terms of the Apache License, Version 2.0 which is available at
  ~ https://www.apache.org/licenses/LICENSE-2.0.
  ~
  ~ Unless required by applicable law or agreed to in writing, software
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
        <title>"""),_display_(/*24.17*/i18n/*24.21*/.getMessage("train.pagetitle")),format.raw/*24.51*/("""</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">

        <link rel="stylesheet" href="/assets/webjars/coreui__coreui/2.1.9/dist/css/coreui.min.css">
        <link rel="stylesheet" href="/assets/css/newstyle.css">


        <script src="/assets/webjars/jquery/3.4.1/dist/jquery.min.js"></script>
        <script src="/assets/webjars/popper.js/1.12.9/dist/umd/popper.min.js"></script>
        <script src="/assets/webjars/bootstrap/4.2.1/dist/js/bootstrap.min.js"></script>
        <script src="/assets/webjars/coreui__coreui/2.1.9/dist/js/coreui.min.js"></script>


        <!-- Icons -->
        <link rel="stylesheet" href="/assets/webjars/coreui__icons/0.3.0/css/coreui-icons.min.css"></script>


        <link rel="shortcut icon" href="/assets/img/favicon.ico">
    </head>

    <body class="app sidebar-show aside-menu-show">
        <header class="app-header navbar">
                <a class="header-text" href="#"><span>"""),_display_(/*46.56*/i18n/*46.60*/.getMessage("train.pagetitle")),format.raw/*46.90*/("""</span></a>
                <div id="sessionSelectDiv" style="display:none; float:right;">
                        <div style="color:white;">"""),_display_(/*48.52*/i18n/*48.56*/.getMessage("train.session.label")),format.raw/*48.90*/("""</div>
                        <select id="sessionSelect" onchange='selectNewSession()'>
                        <option>(Session ID)</option>
                </select>
                </div>
                <div id="workerSelectDiv" style="display:none; float:right">
                        <div style="color:white;">"""),_display_(/*54.52*/i18n/*54.56*/.getMessage("train.session.worker.label")),format.raw/*54.97*/("""</div>
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
                        <li class="nav-item"><a class="nav-link" href="overview"><i class="nav-icon cui-chart"></i>"""),_display_(/*66.117*/i18n/*66.121*/.getMessage("train.nav.overview")),format.raw/*66.154*/("""</a></li>
                        <li class="nav-item"><a class="nav-link" href="model"><i class="nav-icon cui-graph"></i>"""),_display_(/*67.114*/i18n/*67.118*/.getMessage("train.nav.model")),format.raw/*67.148*/("""</a></li>
                        <li class="nav-item"><a class="nav-link" href="system"><i class="nav-icon cui-speedometer"></i>"""),_display_(/*68.121*/i18n/*68.125*/.getMessage("train.nav.system")),format.raw/*68.156*/("""</a></li>
                        <li class="nav-item nav-dropdown">
                            <a class="nav-link nav-dropdown-toggle" href="#">
                                <i class="nav-icon cui-globe"></i> """),_display_(/*71.69*/i18n/*71.73*/.getMessage("train.nav.language")),format.raw/*71.106*/("""
                            """),format.raw/*72.29*/("""</a>
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
                #layers """),format.raw/*89.25*/("""{"""),format.raw/*89.26*/("""
                    """),format.raw/*90.21*/("""height: 725px; /* IE8 */
                    height: 90vh;
                    width: 100%;
                    border: 2px solid #eee;
                """),format.raw/*94.17*/("""}"""),format.raw/*94.18*/("""
                """),format.raw/*95.17*/("""</style>

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
                                        <h2><b>"""),_display_(/*109.49*/i18n/*109.53*/.getMessage("train.model.layerInfoTable.title")),format.raw/*109.100*/("""</b></h2>
                                </div>
                                <div class="box-content">
                                        <table class="table table-bordered table-striped table-condensed" id="layerInfo"></table>
                                </div>
                        </div>

                        <div class="box">
                                <div class="chart-header">
                                        <h2><b>"""),_display_(/*118.49*/i18n/*118.53*/.getMessage("train.overview.chart.updateRatioTitle")),format.raw/*118.105*/("""</b></h2><p id="updateRatioTitleLog10"><b>: log<sub>10</sub></b></p>
                                        <ul class="nav tab-menu nav-tabs" style="position:absolute; margin-top: -36px; right: 27px;">
                                        <li id="mmRatioTab"><a href="javascript:void(0);" onclick="setSelectMeanMagChart('ratios')">"""),_display_(/*120.134*/i18n/*120.138*/.getMessage("train.model.meanmag.btn.ratio")),format.raw/*120.182*/("""</a></li>
                                        <li id="mmParamTab"><a href="javascript:void(0);" onclick="setSelectMeanMagChart('paramMM')">"""),_display_(/*121.135*/i18n/*121.139*/.getMessage("train.model.meanmag.btn.param")),format.raw/*121.183*/("""</a></li>
                                        <li id="mmUpdateTab"><a href="javascript:void(0);" onclick="setSelectMeanMagChart('updateMM')">"""),_display_(/*122.137*/i18n/*122.141*/.getMessage("train.model.meanmag.btn.update")),format.raw/*122.186*/("""</a></li>
                                        </ul>
                                </div>
                                <div class="box-content">
                                        <div id="meanmag" class="center" style="height: 300px;" ></div>
                                        <p id="hoverdata"><span id="updateRatioTitleSmallLog10"><b>log<sub>
                                        10</sub> """),_display_(/*128.51*/i18n/*128.55*/.getMessage("train.overview.chart.updateRatioTitleShort")),format.raw/*128.112*/("""</b></span> <span id="yMeanMagnitudes">0</span>, <b>"""),_display_(/*128.165*/i18n/*128.169*/.getMessage("train.overview.charts.iteration")),format.raw/*128.215*/(""":</b> <span id="xMeanMagnitudes">0</span></p>
                                </div>
                        </div>
                                <div class="box">
                                        <div class="chart-header">
                                        <h2><b>"""),_display_(/*133.49*/i18n/*133.53*/.getMessage("train.model.activationsChart.title")),format.raw/*133.102*/("""</b></h2>
                                </div>
                                <div class="box-content">
                                        <div id="activations" class="center" style="height: 300px;" ></div>
                                        <p id="hoverdata"><b>"""),_display_(/*137.63*/i18n/*137.67*/.getMessage("train.model.activationsChart.titleShort")),format.raw/*137.121*/("""
                                """),format.raw/*138.33*/(""":</b> <span id="yActivations">0</span>
                                        , <b>"""),_display_(/*139.47*/i18n/*139.51*/.getMessage("train.overview.charts.iteration")),format.raw/*139.97*/("""
                                """),format.raw/*140.33*/(""":</b> <span id="xActivations">0</span></p>
                                </div>
                                </div>

                                <div class="box">
                                        <div class="chart-header">
                                        <h2><b>"""),_display_(/*146.49*/i18n/*146.53*/.getMessage("train.model.paramHistChart.title")),format.raw/*146.100*/("""</b></h2>
                                <div id="paramhistSelected" style="float: left"></div>
                                        <div id="paramHistButtonsDiv" style="float: right"></div>
                                        </div>
                                        <div class="box-content">
                                        <div id="parametershistogram" class="center" style="height: 300px;"></div>
                                        </div>
                                        </div>

                                        <div class="box">
                                        <div class="chart-header">
                                        <h2><b>"""),_display_(/*157.49*/i18n/*157.53*/.getMessage("train.model.updateHistChart.title")),format.raw/*157.101*/("""</b></h2>
                                <div id="updatehistSelected" style="float: left"></div>
                                        <div id="updateHistButtonsDiv" style="float: right"></div>
                                        </div>
                                        <div class="box-content">
                                        <div id="updateshistogram" class="center" style="height: 300px;"></div>
                                        </div>
                                        </div>

                                        <div class="box">
                                        <div class="chart-header">
                                        <h2><b>"""),_display_(/*168.49*/i18n/*168.53*/.getMessage("train.model.lrChart.title")),format.raw/*168.93*/("""</b></h2>
                                </div>
                                <div class="box-content">
                                        <div id="learningrate" class="center" style="height: 300px;" ></div>
                                        <p id="hoverdata"><b>"""),_display_(/*172.63*/i18n/*172.67*/.getMessage("train.model.lrChart.titleShort")),format.raw/*172.112*/("""
                                """),format.raw/*173.33*/(""":</b> <span id="yLearningRate">0</span>
                                        , <b>"""),_display_(/*174.47*/i18n/*174.51*/.getMessage("train.overview.charts.iteration")),format.raw/*174.97*/("""
                                """),format.raw/*175.33*/(""":</b> <span id="xLearningRate">0</span></p>
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
               $(document).ready(function () """),format.raw/*253.46*/("""{"""),format.raw/*253.47*/("""
                   """),format.raw/*254.20*/("""renderModelGraph();
                   renderModelPage(true);
               """),format.raw/*256.16*/("""}"""),format.raw/*256.17*/(""");
       </script>

               <!-- Execute periodically (every 2 sec) -->
        <script>
                setInterval(function () """),format.raw/*261.41*/("""{"""),format.raw/*261.42*/("""
                    """),format.raw/*262.21*/("""renderModelPage(false);
                """),format.raw/*263.17*/("""}"""),format.raw/*263.18*/(""", 2000);
        </script>
    </body>
</html>
"""))
      }
    }
  }

  def render(i18n:org.deeplearning4j.ui.api.I18N): play.twirl.api.HtmlFormat.Appendable = apply(i18n)

  def f:((org.deeplearning4j.ui.api.I18N) => play.twirl.api.HtmlFormat.Appendable) = (i18n) => apply(i18n)

  def ref: this.type = this

}


}

/**/
object TrainingModel extends TrainingModel_Scope0.TrainingModel
              /*
                  -- GENERATED --
                  DATE: Tue May 07 19:29:02 AEST 2019
                  SOURCE: c:/DL4J/Git/deeplearning4j/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingModel.scala.html
                  HASH: 5fb101e6fa9971b473edbe3c62197b9e42574345
                  MATRIX: 598->1|731->39|759->41|1677->932|1690->936|1741->966|2756->1954|2769->1958|2820->1988|2991->2132|3004->2136|3059->2170|3412->2496|3425->2500|3487->2541|4023->3049|4037->3053|4092->3086|4244->3210|4258->3214|4310->3244|4469->3375|4483->3379|4536->3410|4781->3628|4794->3632|4849->3665|4907->3695|6347->5107|6376->5108|6426->5130|6610->5286|6639->5287|6685->5305|7249->5841|7263->5845|7333->5892|7827->6358|7841->6362|7916->6414|8283->6752|8298->6756|8365->6800|8539->6945|8554->6949|8621->6993|8797->7140|8812->7144|8880->7189|9329->7610|9343->7614|9423->7671|9505->7724|9520->7728|9589->7774|9903->8060|9917->8064|9989->8113|10298->8394|10312->8398|10389->8452|10452->8486|10566->8572|10580->8576|10648->8622|10711->8656|11032->8949|11046->8953|11116->9000|11846->9702|11860->9706|11931->9754|12660->10455|12674->10459|12736->10499|13046->10781|13060->10785|13128->10830|13191->10864|13306->10951|13320->10955|13388->11001|13451->11035|18027->15582|18057->15583|18107->15604|18215->15683|18245->15684|18416->15826|18446->15827|18497->15849|18567->15890|18597->15891
                  LINES: 20->1|25->1|26->2|48->24|48->24|48->24|70->46|70->46|70->46|72->48|72->48|72->48|78->54|78->54|78->54|90->66|90->66|90->66|91->67|91->67|91->67|92->68|92->68|92->68|95->71|95->71|95->71|96->72|113->89|113->89|114->90|118->94|118->94|119->95|133->109|133->109|133->109|142->118|142->118|142->118|144->120|144->120|144->120|145->121|145->121|145->121|146->122|146->122|146->122|152->128|152->128|152->128|152->128|152->128|152->128|157->133|157->133|157->133|161->137|161->137|161->137|162->138|163->139|163->139|163->139|164->140|170->146|170->146|170->146|181->157|181->157|181->157|192->168|192->168|192->168|196->172|196->172|196->172|197->173|198->174|198->174|198->174|199->175|277->253|277->253|278->254|280->256|280->256|285->261|285->261|286->262|287->263|287->263
                  -- GENERATED --
              */
          