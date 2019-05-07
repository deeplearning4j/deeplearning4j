
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


    <body>
            <!-- Start Header -->
        <div class="navbar">
            <div class="navbar-inner">
                <div class="container-fluid">
                    <a class="btn btn-navbar" data-toggle="collapse" data-target=".top-nav.nav-collapse,.sidebar-nav.nav-collapse">
                        <span class="icon-bar"></span>
                        <span class="icon-bar"></span>
                        <span class="icon-bar"></span>
                    </a>
                    <a class="brand" href="./overview"><span>"""),_display_(/*33.63*/i18n/*33.67*/.getMessage("train.pagetitle")),format.raw/*33.97*/("""</span></a>
                    <div id="sessionSelectDiv" style="display:none; float:right">
                        """),_display_(/*35.26*/i18n/*35.30*/.getMessage("train.session.label")),format.raw/*35.64*/("""
                        """),format.raw/*36.25*/("""<select id="sessionSelect" onchange='selectNewSession()'>
                            <option>(Session ID)</option>
                        </select>
                    </div>
                    <div id="workerSelectDiv" style="display:none; float:right;">
                        """),_display_(/*41.26*/i18n/*41.30*/.getMessage("train.session.worker.label")),format.raw/*41.71*/("""
                        """),format.raw/*42.25*/("""<select id="workerSelect" onchange='selectNewWorker()'>
                            <option>(Worker ID)</option>
                        </select>
                    </div>
                </div>
            </div>
        </div>
            <!-- End Header -->

        <div class="container-fluid-full">
            <div class="row-fluid">

                    <!-- Start Main Menu -->
                <div id="sidebar-left" class="span2">
                    <div class="nav-collapse sidebar-nav">
                        <ul class="nav nav-tabs nav-stacked main-menu">
                            <li><a href="overview"><i class="icon-bar-chart"></i> <span class="hidden-tablet">"""),_display_(/*58.112*/i18n/*58.116*/.getMessage("train.nav.overview")),format.raw/*58.149*/("""</span></a></li>
                            <li class="active"><a href="javascript:void(0);"><i class="icon-tasks"></i> <span class="hidden-tablet">"""),_display_(/*59.134*/i18n/*59.138*/.getMessage("train.nav.model")),format.raw/*59.168*/("""</span></a></li>
                            <li><a href="system"><i class="icon-dashboard"></i> <span class="hidden-tablet">"""),_display_(/*60.110*/i18n/*60.114*/.getMessage("train.nav.system")),format.raw/*60.145*/("""</span></a></li>
                            """),format.raw/*61.161*/("""
                            """),format.raw/*62.29*/("""<li>
                                <a class="dropmenu" href="javascript:void(0);"><i class="icon-folder-close-alt"></i> <span class="hidden-tablet">
                                """),_display_(/*64.34*/i18n/*64.38*/.getMessage("train.nav.language")),format.raw/*64.71*/("""</span></a>
                                <ul>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('en', 'model')"><i class="icon-file-alt"></i> <span class="hidden-tablet">
                                        English</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('de', 'model')"><i class="icon-file-alt"></i> <span class="hidden-tablet">
                                        Deutsch</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ja', 'model')"><i class="icon-file-alt"></i> <span class="hidden-tablet">
                                        日本語</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('zh', 'model')"><i class="icon-file-alt"></i> <span class="hidden-tablet">
                                        中文</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ko', 'model')"><i class="icon-file-alt"></i> <span class="hidden-tablet">
                                        한글</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ru', 'model')"><i class="icon-file-alt"></i> <span class="hidden-tablet">
                                        русский</span></a></li>
                                </ul>
                            </li>
                        </ul>
                    </div>
                </div>
                    <!-- End Main Menu -->

                <noscript>
                    <div class="alert alert-block span10">
                        <h4 class="alert-heading">Warning!</h4>
                        <p>You need to have <a href="http://en.wikipedia.org/wiki/JavaScript" target="_blank">
                            JavaScript</a>
                            enabled to use this site.</p>
                    </div>
                </noscript>

                <style>
                /* Graph */
                #layers """),format.raw/*96.25*/("""{"""),format.raw/*96.26*/("""
                    """),format.raw/*97.21*/("""height: 725px; /* IE8 */
                    height: 90vh;
                    width: 100%;
                    border: 2px solid #eee;
                """),format.raw/*101.17*/("""}"""),format.raw/*101.18*/("""
                """),format.raw/*102.17*/("""</style>

                    <!-- Start Content -->
                <div id="content" class="span10">

                    <div class="row-fluid span5">
                        <div id="layers"></div>
                    </div>
                        <!-- Start Layer Details -->
                    <div class="row-fluid span7" id="layerDetails">

                        <div class="box">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*115.41*/i18n/*115.45*/.getMessage("train.model.layerInfoTable.title")),format.raw/*115.92*/("""</b></h2>
                            </div>
                            <div class="box-content">
                                <table class="table table-bordered table-striped table-condensed" id="layerInfo"></table>
                            </div>
                        </div>

                        <div class="box">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*124.41*/i18n/*124.45*/.getMessage("train.overview.chart.updateRatioTitle")),format.raw/*124.97*/("""</b></h2><p id="updateRatioTitleLog10"><b>: log<sub>10</sub></b></p>
                                <ul class="nav tab-menu nav-tabs" style="position:absolute; margin-top: -36px; right: 27px;">
                                    <li id="mmRatioTab"><a href="javascript:void(0);" onclick="setSelectMeanMagChart('ratios')">"""),_display_(/*126.130*/i18n/*126.134*/.getMessage("train.model.meanmag.btn.ratio")),format.raw/*126.178*/("""</a></li>
                                    <li id="mmParamTab"><a href="javascript:void(0);" onclick="setSelectMeanMagChart('paramMM')">"""),_display_(/*127.131*/i18n/*127.135*/.getMessage("train.model.meanmag.btn.param")),format.raw/*127.179*/("""</a></li>
                                    <li id="mmUpdateTab"><a href="javascript:void(0);" onclick="setSelectMeanMagChart('updateMM')">"""),_display_(/*128.133*/i18n/*128.137*/.getMessage("train.model.meanmag.btn.update")),format.raw/*128.182*/("""</a></li>
                                </ul>
                            </div>
                            <div class="box-content">
                                <div id="meanmag" class="center" style="height: 300px;" ></div>
                                <p id="hoverdata"><span id="updateRatioTitleSmallLog10"><b>log<sub>
                                    10</sub> """),_display_(/*134.47*/i18n/*134.51*/.getMessage("train.overview.chart.updateRatioTitleShort")),format.raw/*134.108*/("""</b></span> <span id="yMeanMagnitudes">
                                    0</span>
                                    , <b>"""),_display_(/*136.43*/i18n/*136.47*/.getMessage("train.overview.charts.iteration")),format.raw/*136.93*/("""
                                        """),format.raw/*137.41*/(""":</b> <span id="xMeanMagnitudes">0</span></p>
                            </div>
                        </div>

                        <div class="box">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*143.41*/i18n/*143.45*/.getMessage("train.model.activationsChart.title")),format.raw/*143.94*/("""</b></h2>
                            </div>
                            <div class="box-content">
                                <div id="activations" class="center" style="height: 300px;" ></div>
                                <p id="hoverdata"><b>"""),_display_(/*147.55*/i18n/*147.59*/.getMessage("train.model.activationsChart.titleShort")),format.raw/*147.113*/("""
                                    """),format.raw/*148.37*/(""":</b> <span id="yActivations">0</span>
                                    , <b>"""),_display_(/*149.43*/i18n/*149.47*/.getMessage("train.overview.charts.iteration")),format.raw/*149.93*/("""
                                        """),format.raw/*150.41*/(""":</b> <span id="xActivations">0</span></p>
                            </div>
                        </div>

                        <div class="box">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*156.41*/i18n/*156.45*/.getMessage("train.model.paramHistChart.title")),format.raw/*156.92*/("""</b></h2>
                                <div id="paramhistSelected" style="float: left"></div>
                                <div id="paramHistButtonsDiv" style="float: right"></div>
                            </div>
                            <div class="box-content">
                                <div id="parametershistogram" class="center" style="height: 300px;"></div>
                            </div>
                        </div>

                        <div class="box">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*167.41*/i18n/*167.45*/.getMessage("train.model.updateHistChart.title")),format.raw/*167.93*/("""</b></h2>
                                <div id="updatehistSelected" style="float: left"></div>
                                <div id="updateHistButtonsDiv" style="float: right"></div>
                            </div>
                            <div class="box-content">
                                <div id="updateshistogram" class="center" style="height: 300px;"></div>
                            </div>
                        </div>

                        <div class="box">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*178.41*/i18n/*178.45*/.getMessage("train.model.lrChart.title")),format.raw/*178.85*/("""</b></h2>
                            </div>
                            <div class="box-content">
                                <div id="learningrate" class="center" style="height: 300px;" ></div>
                                <p id="hoverdata"><b>"""),_display_(/*182.55*/i18n/*182.59*/.getMessage("train.model.lrChart.titleShort")),format.raw/*182.104*/("""
                                    """),format.raw/*183.37*/(""":</b> <span id="yLearningRate">0</span>
                                    , <b>"""),_display_(/*184.43*/i18n/*184.47*/.getMessage("train.overview.charts.iteration")),format.raw/*184.93*/("""
                                        """),format.raw/*185.41*/(""":</b> <span id="xLearningRate">0</span></p>
                            </div>
                        </div>

                    </div>
                        <!-- End Layer Details-->

                        <!-- Begin Zero State -->
                    <div class="row-fluid span6" id="zeroState">
                        <div class="box">
                            <div class="box-header">
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
                    <!-- End Content -->
            </div> <!-- End Container -->
        </div> <!-- End Row Fluid-->

        <!-- Start JavaScript-->
        <script src="/assets/webjars/jquery/2.2.0/jquery.min.js"></script>
        <script src="/assets/webjars/jquery-ui/1.10.2/ui/minified/jquery-ui.min.js"></script>
        <script src="/assets/webjars/jquery-migrate/1.2.1/jquery-migrate.min.js"></script>
        <script src="/assets/webjars/jquery-ui-touch-punch/0.2.2/jquery.ui.touch-punch.min.js"></script>
        <script src="/assets/webjars/modernizr/2.8.3/modernizr.min.js"></script>
        <script src="/assets/webjars/bootstrap/2.3.2/js/bootstrap.min.js"></script>
        <script src="/assets/webjars/jquery-cookie/1.4.1-1/jquery.cookie.js"></script>
        <script src="/assets/webjars/fullcalendar/1.6.4/fullcalendar.min.js"></script>
        <script src="/assets/webjars/datatables/1.9.4/media/js/jquery.dataTables.min.js"></script>
        <script src="/assets/webjars/excanvas/3/excanvas.js"></script>
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
               $(document).ready(function () """),format.raw/*266.46*/("""{"""),format.raw/*266.47*/("""
                   """),format.raw/*267.20*/("""renderModelGraph();
                   renderModelPage(true);
               """),format.raw/*269.16*/("""}"""),format.raw/*269.17*/(""");
       </script>

               <!-- Execute periodically (every 2 sec) -->
        <script>
                setInterval(function () """),format.raw/*274.41*/("""{"""),format.raw/*274.42*/("""
                    """),format.raw/*275.21*/("""renderModelPage(false);
                """),format.raw/*276.17*/("""}"""),format.raw/*276.18*/(""", 2000);
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
                  DATE: Tue May 07 18:30:26 AEST 2019
                  SOURCE: c:/DL4J/Git/deeplearning4j/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingModel.scala.html
                  HASH: 7b4439f06a6d791a1ea2f41279876cb4d3f98cb9
                  MATRIX: 598->1|731->39|759->41|2172->1427|2185->1431|2236->1461|2384->1582|2397->1586|2452->1620|2506->1646|2822->1935|2835->1939|2897->1980|2951->2006|3680->2707|3694->2711|3749->2744|3928->2895|3942->2899|3994->2929|4149->3056|4163->3060|4216->3091|4291->3269|4349->3299|4562->3485|4575->3489|4629->3522|6908->5773|6937->5774|6987->5796|7172->5952|7202->5953|7249->5971|7776->6470|7790->6474|7859->6521|8319->6953|8333->6957|8407->7009|8762->7335|8777->7339|8844->7383|9014->7524|9029->7528|9096->7572|9268->7715|9283->7719|9351->7764|9764->8149|9778->8153|9858->8210|10015->8339|10029->8343|10097->8389|10168->8431|10450->8685|10464->8689|10535->8738|10820->8995|10834->8999|10911->9053|10978->9091|11088->9173|11102->9177|11170->9223|11241->9265|11520->9516|11534->9520|11603->9567|12227->10163|12241->10167|12311->10215|12934->10810|12948->10814|13010->10854|13296->11112|13310->11116|13378->11161|13445->11199|13556->11282|13570->11286|13638->11332|13709->11374|18907->16543|18937->16544|18987->16565|19095->16644|19125->16645|19296->16787|19326->16788|19377->16810|19447->16851|19477->16852
                  LINES: 20->1|25->1|26->2|57->33|57->33|57->33|59->35|59->35|59->35|60->36|65->41|65->41|65->41|66->42|82->58|82->58|82->58|83->59|83->59|83->59|84->60|84->60|84->60|85->61|86->62|88->64|88->64|88->64|120->96|120->96|121->97|125->101|125->101|126->102|139->115|139->115|139->115|148->124|148->124|148->124|150->126|150->126|150->126|151->127|151->127|151->127|152->128|152->128|152->128|158->134|158->134|158->134|160->136|160->136|160->136|161->137|167->143|167->143|167->143|171->147|171->147|171->147|172->148|173->149|173->149|173->149|174->150|180->156|180->156|180->156|191->167|191->167|191->167|202->178|202->178|202->178|206->182|206->182|206->182|207->183|208->184|208->184|208->184|209->185|290->266|290->266|291->267|293->269|293->269|298->274|298->274|299->275|300->276|300->276
                  -- GENERATED --
              */
          