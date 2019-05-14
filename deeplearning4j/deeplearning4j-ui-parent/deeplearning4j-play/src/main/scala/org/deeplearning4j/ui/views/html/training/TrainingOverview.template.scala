
/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.ui.views.html.training

import play.twirl.api._
import play.twirl.api.TemplateMagic._


     object TrainingOverview_Scope0 {
import models._
import controllers._
import play.api.i18n._
import views.html._
import play.api.templates.PlayMagic._
import play.api.mvc._
import play.api.data._

class TrainingOverview extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template1[org.deeplearning4j.ui.api.I18N,play.twirl.api.HtmlFormat.Appendable] {

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
        <div class="app-body">
            <div class="sidebar">
                <nav class="sidebar-nav">
                    <ul class="nav">
                        <li class="nav-item"><a class="nav-link" href="overview"><i class="nav-icon cui-chart"></i>"""),_display_(/*64.117*/i18n/*64.121*/.getMessage("train.nav.overview")),format.raw/*64.154*/("""</a></li>
                        <li class="nav-item"><a class="nav-link" href="model"><i class="nav-icon cui-graph"></i>"""),_display_(/*65.114*/i18n/*65.118*/.getMessage("train.nav.model")),format.raw/*65.148*/("""</a></li>
                        <li class="nav-item"><a class="nav-link" href="system"><i class="nav-icon cui-speedometer"></i>"""),_display_(/*66.121*/i18n/*66.125*/.getMessage("train.nav.system")),format.raw/*66.156*/("""</a></li>
                        <li class="nav-item nav-dropdown">
                            <a class="nav-link nav-dropdown-toggle" href="#">
                                <i class="nav-icon cui-globe"></i> """),_display_(/*69.69*/i18n/*69.73*/.getMessage("train.nav.language")),format.raw/*69.106*/("""
                            """),format.raw/*70.29*/("""</a>
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
            <main id="content" class="main">
                <div class="row">

                    <div class="col-8 chart-box">
                        <div class="chart-header">
                            <h2><b>"""),_display_(/*89.37*/i18n/*89.41*/.getMessage("train.overview.chart.scoreTitle")),format.raw/*89.87*/("""</b></h2>
                        </div>
                        <div>
                            <div id="scoreiterchart" class="center" style="height: 300px;" ></div>
                            <p id="hoverdata"><b>"""),_display_(/*93.51*/i18n/*93.55*/.getMessage("train.overview.chart.scoreTitleShort")),format.raw/*93.106*/("""
                                """),format.raw/*94.33*/(""":</b> <span id="y">0</span>, <b>"""),_display_(/*94.66*/i18n/*94.70*/.getMessage("train.overview.charts.iteration")),format.raw/*94.116*/("""
                                """),format.raw/*95.33*/(""":</b> <span id="x">
                                0</span></p>
                        </div>
                    </div>
                        <!-- End Score Chart-->
                        <!-- Start Model Table-->
                    <div class="col-4 chart-box">
                        <div class="chart-header">
                            <h2><b>"""),_display_(/*103.37*/i18n/*103.41*/.getMessage("train.overview.perftable.title")),format.raw/*103.86*/("""</b></h2>
                        </div>
                        <div>
                            <table class="table table-bordered table-striped table-condensed">
                                <tr>
                                    <td>"""),_display_(/*108.42*/i18n/*108.46*/.getMessage("train.overview.modeltable.modeltype")),format.raw/*108.96*/("""</td>
                                    <td id="modelType">Loading...</td>
                                </tr>
                                <tr>
                                    <td>"""),_display_(/*112.42*/i18n/*112.46*/.getMessage("train.overview.modeltable.nLayers")),format.raw/*112.94*/("""</td>
                                    <td id="nLayers">Loading...</td>
                                </tr>
                                <tr>
                                    <td>"""),_display_(/*116.42*/i18n/*116.46*/.getMessage("train.overview.modeltable.nParams")),format.raw/*116.94*/("""</td>
                                    <td id="nParams">Loading...</td>
                                </tr>
                                <tr>
                                    <td>"""),_display_(/*120.42*/i18n/*120.46*/.getMessage("train.overview.perftable.startTime")),format.raw/*120.95*/("""</td>
                                    <td id="startTime">Loading...</td>
                                </tr>
                                <tr>
                                    <td>"""),_display_(/*124.42*/i18n/*124.46*/.getMessage("train.overview.perftable.totalRuntime")),format.raw/*124.98*/("""</td>
                                    <td id="totalRuntime">Loading...</td>
                                </tr>
                                <tr>
                                    <td>"""),_display_(/*128.42*/i18n/*128.46*/.getMessage("train.overview.perftable.lastUpdate")),format.raw/*128.96*/("""</td>
                                    <td id="lastUpdate">Loading...</td>
                                </tr>
                                <tr>
                                    <td>"""),_display_(/*132.42*/i18n/*132.46*/.getMessage("train.overview.perftable.totalParamUpdates")),format.raw/*132.103*/("""</td>
                                    <td id="totalParamUpdates">Loading...</td>
                                </tr>
                                <tr>
                                    <td>"""),_display_(/*136.42*/i18n/*136.46*/.getMessage("train.overview.perftable.updatesPerSec")),format.raw/*136.99*/("""</td>
                                    <td id="updatesPerSec">Loading...</td>
                                </tr>
                                <tr>
                                    <td>"""),_display_(/*140.42*/i18n/*140.46*/.getMessage("train.overview.perftable.examplesPerSec")),format.raw/*140.100*/("""</td>
                                    <td id="examplesPerSec">Loading...</td>
                                </tr>
                            </table>
                        </div>
                    </div>
                        <!--End Model Table -->
                </div>


                <div class="row">
                        <!--Start Ratio Table -->
                    <div class="col chart-box">
                        <div class="chart-header">
                            <h2><b>"""),_display_(/*154.37*/i18n/*154.41*/.getMessage("train.overview.chart.updateRatioTitle")),format.raw/*154.93*/(""": log<sub>10</sub></b></h2>
                        </div>
                        <div class="box-content">
                            <div id="updateRatioChart" class="center" style="height: 300px;" ></div>
                            <p id="hoverdata"><b>"""),_display_(/*158.51*/i18n/*158.55*/.getMessage("train.overview.chart.updateRatioTitleShort")),format.raw/*158.112*/("""
                                """),format.raw/*159.33*/(""":</b> <span id="yRatio">0</span>, <b>log<sub>
                                10</sub> """),_display_(/*160.43*/i18n/*160.47*/.getMessage("train.overview.chart.updateRatioTitleShort")),format.raw/*160.104*/("""
                                """),format.raw/*161.33*/(""":</b> <span id="yLogRatio">0</span>
                                , <b>"""),_display_(/*162.39*/i18n/*162.43*/.getMessage("train.overview.charts.iteration")),format.raw/*162.89*/(""":</b> <span id="xRatio">
                                    0</span></p>
                        </div>

                    </div>
                        <!--End Ratio Table -->
                        <!--Start Variance Table -->
                    <div class="col chart-box">
                        <div class="chart-header">
                            <h2><b>"""),_display_(/*171.37*/i18n/*171.41*/.getMessage("train.overview.chart.stdevTitle")),format.raw/*171.87*/(""": log<sub>10</sub></b></h2>
                            <ul class="nav tab-menu nav-tabs" style="position:absolute; margin-top: -30px; right: 22px;">
                                <li class="active" id="stdevActivations"><a href="javascript:void(0);" onclick="selectStdevChart('stdevActivations')">"""),_display_(/*173.152*/i18n/*173.156*/.getMessage("train.overview.chart.stdevBtn.activations")),format.raw/*173.212*/("""</a></li>
                                <li id="stdevGradients"><a href="javascript:void(0);" onclick="selectStdevChart('stdevGradients')">"""),_display_(/*174.133*/i18n/*174.137*/.getMessage("train.overview.chart.stdevBtn.gradients")),format.raw/*174.191*/("""</a></li>
                                <li id="stdevUpdates"><a href="javascript:void(0);" onclick="selectStdevChart('stdevUpdates')">"""),_display_(/*175.129*/i18n/*175.133*/.getMessage("train.overview.chart.stdevBtn.updates")),format.raw/*175.185*/("""</a></li>
                            </ul>
                        </div>
                        <div class="box-content">
                            <div id="stdevChart" class="center" style="height: 300px;" ></div>
                            <p id="hoverdata"><b>"""),_display_(/*180.51*/i18n/*180.55*/.getMessage("train.overview.chart.stdevTitleShort")),format.raw/*180.106*/("""
                                """),format.raw/*181.33*/(""":</b> <span id="yStdev">0</span>, <b>log<sub>
                                10</sub> """),_display_(/*182.43*/i18n/*182.47*/.getMessage("train.overview.chart.stdevTitleShort")),format.raw/*182.98*/("""
                                """),format.raw/*183.33*/(""":</b> <span id="yLogStdev">0</span>
                                , <b>"""),_display_(/*184.39*/i18n/*184.43*/.getMessage("train.overview.charts.iteration")),format.raw/*184.89*/(""":</b> <span id="xStdev">
                                    0</span></p>
                        </div>
                    </div>
                        <!-- End Variance Table -->
                </div>
            </main>
        </div>

        <script src="/assets/webjars/fullcalendar/1.6.4/fullcalendar.min.js"></script>
        <script src="/assets/webjars/excanvas/3/excanvas.js"></script>
        <script src="/assets/webjars/retinajs/0.0.2/retina.js"></script>
        <script src="/assets/webjars/flot/0.8.3/jquery.flot.js"></script>
        <script src="/assets/webjars/flot/0.8.3/jquery.flot.pie.js"></script>
        <script src="/assets/webjars/flot/0.8.3/jquery.flot.stack.js"></script>
        <script src="/assets/webjars/flot/0.8.3/jquery.flot.resize.min.js"></script>
        <script src="/assets/webjars/flot/0.8.3/jquery.flot.selection.js"></script>

        <script src="/assets/js/train/overview.js"></script>    <!-- Charts and tables are generated here! -->
        <script src="/assets/js/train/train.js"></script>   <!-- Common (lang selection, etc) -->
        <script src="/assets/js/counter.js"></script>

        <!-- Execute once on page load -->
        <script>
                $(document).ready(function () """),format.raw/*208.47*/("""{"""),format.raw/*208.48*/("""
                    """),format.raw/*209.21*/("""renderOverviewPage(true);
                """),format.raw/*210.17*/("""}"""),format.raw/*210.18*/(""");
        </script>

        <!-- Execute periodically (every 2 sec) -->
        <script>
                setInterval(function () """),format.raw/*215.41*/("""{"""),format.raw/*215.42*/("""
                    """),format.raw/*216.21*/("""renderOverviewPage(false);
                """),format.raw/*217.17*/("""}"""),format.raw/*217.18*/(""", 2000);
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
object TrainingOverview extends TrainingOverview_Scope0.TrainingOverview
              /*
                  -- GENERATED --
                  DATE: Tue May 07 21:39:41 AEST 2019
                  SOURCE: c:/DL4J/Git/deeplearning4j/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingOverview.scala.html
                  HASH: 2c9149a6ec8ab4fcc1b316e6ecd5befab2c45af1
                  MATRIX: 604->1|737->39|765->41|1683->932|1696->936|1747->966|2759->1951|2772->1955|2823->1985|2994->2129|3007->2133|3062->2167|3415->2493|3428->2497|3490->2538|3995->3015|4009->3019|4064->3052|4216->3176|4230->3180|4282->3210|4441->3341|4455->3345|4508->3376|4753->3594|4766->3598|4821->3631|4879->3661|6460->5215|6473->5219|6540->5265|6791->5489|6804->5493|6877->5544|6939->5578|6999->5611|7012->5615|7080->5661|7142->5695|7536->6061|7550->6065|7617->6110|7894->6359|7908->6363|7980->6413|8205->6610|8219->6614|8289->6662|8512->6857|8526->6861|8596->6909|8819->7104|8833->7108|8904->7157|9129->7354|9143->7358|9217->7410|9445->7610|9459->7614|9531->7664|9757->7862|9771->7866|9851->7923|10084->8128|10098->8132|10173->8185|10402->8386|10416->8390|10493->8444|11042->8965|11056->8969|11130->9021|11422->9285|11436->9289|11516->9346|11579->9380|11696->9469|11710->9473|11790->9530|11853->9564|11956->9639|11970->9643|12038->9689|12444->10067|12458->10071|12526->10117|12858->10420|12873->10424|12952->10480|13124->10623|13139->10627|13216->10681|13384->10820|13399->10824|13474->10876|13777->11151|13791->11155|13865->11206|13928->11240|14045->11329|14059->11333|14132->11384|14195->11418|14298->11493|14312->11497|14380->11543|15679->12813|15709->12814|15760->12836|15832->12879|15862->12880|16027->13016|16057->13017|16108->13039|16181->13083|16211->13084
                  LINES: 20->1|25->1|26->2|48->24|48->24|48->24|70->46|70->46|70->46|72->48|72->48|72->48|78->54|78->54|78->54|88->64|88->64|88->64|89->65|89->65|89->65|90->66|90->66|90->66|93->69|93->69|93->69|94->70|113->89|113->89|113->89|117->93|117->93|117->93|118->94|118->94|118->94|118->94|119->95|127->103|127->103|127->103|132->108|132->108|132->108|136->112|136->112|136->112|140->116|140->116|140->116|144->120|144->120|144->120|148->124|148->124|148->124|152->128|152->128|152->128|156->132|156->132|156->132|160->136|160->136|160->136|164->140|164->140|164->140|178->154|178->154|178->154|182->158|182->158|182->158|183->159|184->160|184->160|184->160|185->161|186->162|186->162|186->162|195->171|195->171|195->171|197->173|197->173|197->173|198->174|198->174|198->174|199->175|199->175|199->175|204->180|204->180|204->180|205->181|206->182|206->182|206->182|207->183|208->184|208->184|208->184|232->208|232->208|233->209|234->210|234->210|239->215|239->215|240->216|241->217|241->217
                  -- GENERATED --
              */
          