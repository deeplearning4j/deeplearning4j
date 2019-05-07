
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
        <link rel="stylesheet" href="/assets/css/newstyle.css">


        <script src="http://code.jquery.com/jquery-3.3.1.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js"></script>
        <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.1.1/js/bootstrap.min.js"></script>
        <script src="/assets/webjars/coreui__coreui/2.1.9/dist/js/coreui.min.js"></script>

        <link rel="shortcut icon" href="/assets/img/favicon.ico">
    </head>

    <body class="app sidebar-show aside-menu-show">
        <header class="app-header navbar">
            <div class="navbar">
                <div class="navbar-inner">
                    <div class="container-fluid">
                        <a class="btn btn-navbar" data-toggle="collapse" data-target=".top-nav.nav-collapse,.sidebar-nav.nav-collapse">
                            <span class="icon-bar"></span>
                            <span class="icon-bar"></span>
                            <span class="icon-bar"></span>
                        </a>
                        <a class="brand" href="#"><span>"""),_display_(/*49.58*/i18n/*49.62*/.getMessage("train.pagetitle")),format.raw/*49.92*/("""</span></a>
                        <div id="sessionSelectDiv" style="display:none; float:right">
                            """),_display_(/*51.30*/i18n/*51.34*/.getMessage("train.session.label")),format.raw/*51.68*/("""
                            """),format.raw/*52.29*/("""<select id="sessionSelect" onchange='selectNewSession()'>
                                <option>(Session ID)</option>
                            </select>
                        </div>
                        <div id="workerSelectDiv" style="display:none; float:right;">
                            """),_display_(/*57.30*/i18n/*57.34*/.getMessage("train.session.worker.label")),format.raw/*57.75*/("""
                            """),format.raw/*58.29*/("""<select id="workerSelect" onchange='selectNewWorker()'>
                                <option>(Worker ID)</option>
                            </select>
                        </div>
                    </div>
                </div>
            </div>
        </header>
        <div class="app-body">
            <div class="sidebar">
                <nav class="sidebar-nav">
                    <ul class="nav">
                        <li class="nav-item"><a class="nav-link" href="javascript:void(0);"><i class="cui-chart"></i>"""),_display_(/*70.119*/i18n/*70.123*/.getMessage("train.nav.overview")),format.raw/*70.156*/("""</a></li>
                        <li class="nav-item"><a class="nav-link" href="model"><i class="icon-tasks"></i>"""),_display_(/*71.106*/i18n/*71.110*/.getMessage("train.nav.model")),format.raw/*71.140*/("""</a></li>
                        <li class="nav-item"><a class="nav-link" href="system"><i class="icon-dashboard"></i>"""),_display_(/*72.111*/i18n/*72.115*/.getMessage("train.nav.system")),format.raw/*72.146*/("""</a></li>
                        <li class="nav-item nav-dropdown">
                            <a class="nav-link nav-dropdown-toggle" href="#">
                                <i class="nav-icon cui-puzzle"></i> """),_display_(/*75.70*/i18n/*75.74*/.getMessage("train.nav.language")),format.raw/*75.107*/("""
                            """),format.raw/*76.29*/("""</a>
                            <ul class="nav-dropdown-items">
                                <li class="nav-item"><a class="submenu" href="javascript:void(0);" onclick="languageSelect('en', 'overview')"><i class="icon-file-alt"></i>English</a></li>
                                <li class="nav-item"><a class="submenu" href="javascript:void(0);" onclick="languageSelect('de', 'overview')"><i class="icon-file-alt"></i>Deutsch</a></li>
                                <li class="nav-item"><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ja', 'overview')"><i class="icon-file-alt"></i>日本語</a></li>
                                <li class="nav-item"><a class="submenu" href="javascript:void(0);" onclick="languageSelect('zh', 'overview')"><i class="icon-file-alt"></i>中文</a></li>
                                <li class="nav-item"><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ko', 'overview')"><i class="icon-file-alt"></i>한글</a></li>
                                <li class="nav-item"><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ru', 'overview')"><i class="icon-file-alt"></i>русский</a></li>
                            </ul>
                        </li>
                    </ul>

                </nav>
            </div>
            <main id="content" class="main">
                <div class="row">

                    <div class="col-8">
                        <div>
                            <h2><b>"""),_display_(/*95.37*/i18n/*95.41*/.getMessage("train.overview.chart.scoreTitle")),format.raw/*95.87*/("""</b></h2>
                        </div>
                        <div>
                            <div id="scoreiterchart" class="center" style="height: 300px;" ></div>
                            <p id="hoverdata"><b>"""),_display_(/*99.51*/i18n/*99.55*/.getMessage("train.overview.chart.scoreTitleShort")),format.raw/*99.106*/("""
                                """),format.raw/*100.33*/(""":</b> <span id="y">0</span>, <b>"""),_display_(/*100.66*/i18n/*100.70*/.getMessage("train.overview.charts.iteration")),format.raw/*100.116*/("""
                                """),format.raw/*101.33*/(""":</b> <span id="x">
                                0</span></p>
                        </div>
                    </div>
                        <!-- End Score Chart-->
                        <!-- Start Model Table-->
                    <div class="col-4">
                        <div>
                            <h2><b>"""),_display_(/*109.37*/i18n/*109.41*/.getMessage("train.overview.perftable.title")),format.raw/*109.86*/("""</b></h2>
                        </div>
                        <div>
                            <table class="table table-bordered table-striped table-condensed">
                                <tr>
                                    <td>"""),_display_(/*114.42*/i18n/*114.46*/.getMessage("train.overview.modeltable.modeltype")),format.raw/*114.96*/("""</td>
                                    <td id="modelType">Loading...</td>
                                </tr>
                                <tr>
                                    <td>"""),_display_(/*118.42*/i18n/*118.46*/.getMessage("train.overview.modeltable.nLayers")),format.raw/*118.94*/("""</td>
                                    <td id="nLayers">Loading...</td>
                                </tr>
                                <tr>
                                    <td>"""),_display_(/*122.42*/i18n/*122.46*/.getMessage("train.overview.modeltable.nParams")),format.raw/*122.94*/("""</td>
                                    <td id="nParams">Loading...</td>
                                </tr>
                                <tr>
                                    <td>"""),_display_(/*126.42*/i18n/*126.46*/.getMessage("train.overview.perftable.startTime")),format.raw/*126.95*/("""</td>
                                    <td id="startTime">Loading...</td>
                                </tr>
                                <tr>
                                    <td>"""),_display_(/*130.42*/i18n/*130.46*/.getMessage("train.overview.perftable.totalRuntime")),format.raw/*130.98*/("""</td>
                                    <td id="totalRuntime">Loading...</td>
                                </tr>
                                <tr>
                                    <td>"""),_display_(/*134.42*/i18n/*134.46*/.getMessage("train.overview.perftable.lastUpdate")),format.raw/*134.96*/("""</td>
                                    <td id="lastUpdate">Loading...</td>
                                </tr>
                                <tr>
                                    <td>"""),_display_(/*138.42*/i18n/*138.46*/.getMessage("train.overview.perftable.totalParamUpdates")),format.raw/*138.103*/("""</td>
                                    <td id="totalParamUpdates">Loading...</td>
                                </tr>
                                <tr>
                                    <td>"""),_display_(/*142.42*/i18n/*142.46*/.getMessage("train.overview.perftable.updatesPerSec")),format.raw/*142.99*/("""</td>
                                    <td id="updatesPerSec">Loading...</td>
                                </tr>
                                <tr>
                                    <td>"""),_display_(/*146.42*/i18n/*146.46*/.getMessage("train.overview.perftable.examplesPerSec")),format.raw/*146.100*/("""</td>
                                    <td id="examplesPerSec">Loading...</td>
                                </tr>
                            </table>
                        </div>
                    </div>
                        <!--End Model Table -->
                </div>


                <div class="row">
                        <!--Start Ratio Table -->
                    <div class="col">
                        <div class="box-header">
                            <h2><b>"""),_display_(/*160.37*/i18n/*160.41*/.getMessage("train.overview.chart.updateRatioTitle")),format.raw/*160.93*/(""": log<sub>10</sub></b></h2>
                        </div>
                        <div class="box-content">
                            <div id="updateRatioChart" class="center" style="height: 300px;" ></div>
                            <p id="hoverdata"><b>"""),_display_(/*164.51*/i18n/*164.55*/.getMessage("train.overview.chart.updateRatioTitleShort")),format.raw/*164.112*/("""
                                """),format.raw/*165.33*/(""":</b> <span id="yRatio">0</span>, <b>log<sub>
                                10</sub> """),_display_(/*166.43*/i18n/*166.47*/.getMessage("train.overview.chart.updateRatioTitleShort")),format.raw/*166.104*/("""
                                """),format.raw/*167.33*/(""":</b> <span id="yLogRatio">0</span>
                                , <b>"""),_display_(/*168.39*/i18n/*168.43*/.getMessage("train.overview.charts.iteration")),format.raw/*168.89*/(""":</b> <span id="xRatio">
                                    0</span></p>
                        </div>
                    </div>
                        <!--End Ratio Table -->
                        <!--Start Variance Table -->
                    <div class="col">
                        <div class="box-header">
                            <h2><b>"""),_display_(/*176.37*/i18n/*176.41*/.getMessage("train.overview.chart.stdevTitle")),format.raw/*176.87*/(""": log<sub>10</sub></b></h2>
                            <ul class="nav tab-menu nav-tabs" style="position:absolute; margin-top: -11px; right: 22px;">
                                <li class="active" id="stdevActivations"><a href="javascript:void(0);" onclick="selectStdevChart('stdevActivations')">"""),_display_(/*178.152*/i18n/*178.156*/.getMessage("train.overview.chart.stdevBtn.activations")),format.raw/*178.212*/("""</a></li>
                                <li id="stdevGradients"><a href="javascript:void(0);" onclick="selectStdevChart('stdevGradients')">"""),_display_(/*179.133*/i18n/*179.137*/.getMessage("train.overview.chart.stdevBtn.gradients")),format.raw/*179.191*/("""</a></li>
                                <li id="stdevUpdates"><a href="javascript:void(0);" onclick="selectStdevChart('stdevUpdates')">"""),_display_(/*180.129*/i18n/*180.133*/.getMessage("train.overview.chart.stdevBtn.updates")),format.raw/*180.185*/("""</a></li>
                            </ul>
                        </div>
                        <div class="box-content">
                            <div id="stdevChart" class="center" style="height: 300px;" ></div>
                            <p id="hoverdata"><b>"""),_display_(/*185.51*/i18n/*185.55*/.getMessage("train.overview.chart.stdevTitleShort")),format.raw/*185.106*/("""
                                """),format.raw/*186.33*/(""":</b> <span id="yStdev">0</span>, <b>log<sub>
                                10</sub> """),_display_(/*187.43*/i18n/*187.47*/.getMessage("train.overview.chart.stdevTitleShort")),format.raw/*187.98*/("""
                                """),format.raw/*188.33*/(""":</b> <span id="yLogStdev">0</span>
                                , <b>"""),_display_(/*189.39*/i18n/*189.43*/.getMessage("train.overview.charts.iteration")),format.raw/*189.89*/(""":</b> <span id="xStdev">
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
                $(document).ready(function () """),format.raw/*213.47*/("""{"""),format.raw/*213.48*/("""
                    """),format.raw/*214.21*/("""renderOverviewPage(true);
                """),format.raw/*215.17*/("""}"""),format.raw/*215.18*/(""");
        </script>

        <!-- Execute periodically (every 2 sec) -->
        <script>
                setInterval(function () """),format.raw/*220.41*/("""{"""),format.raw/*220.42*/("""
                    """),format.raw/*221.21*/("""renderOverviewPage(false);
                """),format.raw/*222.17*/("""}"""),format.raw/*222.18*/(""", 2000);
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
                  DATE: Tue May 07 15:42:41 AEST 2019
                  SOURCE: c:/DL4J/Git/deeplearning4j/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingOverview.scala.html
                  HASH: 5f049056249e497fff3d9ece524ad5bdccc05afa
                  MATRIX: 604->1|737->39|765->41|1683->932|1696->936|1747->966|3127->2319|3140->2323|3191->2353|3347->2482|3360->2486|3415->2520|3473->2550|3809->2859|3822->2863|3884->2904|3942->2934|4517->3481|4531->3485|4586->3518|4730->3634|4744->3638|4796->3668|4945->3789|4959->3793|5012->3824|5258->4043|5271->4047|5326->4080|5384->4110|6928->5627|6941->5631|7008->5677|7259->5901|7272->5905|7345->5956|7408->5990|7469->6023|7483->6027|7552->6073|7615->6107|7978->6442|7992->6446|8059->6491|8336->6740|8350->6744|8422->6794|8647->6991|8661->6995|8731->7043|8954->7238|8968->7242|9038->7290|9261->7485|9275->7489|9346->7538|9571->7735|9585->7739|9659->7791|9887->7991|9901->7995|9973->8045|10199->8243|10213->8247|10293->8304|10526->8509|10540->8513|10615->8566|10844->8767|10858->8771|10935->8825|11472->9334|11486->9338|11560->9390|11852->9654|11866->9658|11946->9715|12009->9749|12126->9838|12140->9842|12220->9899|12283->9933|12386->10008|12400->10012|12468->10058|12860->10422|12874->10426|12942->10472|13274->10775|13289->10779|13368->10835|13540->10978|13555->10982|13632->11036|13800->11175|13815->11179|13890->11231|14193->11506|14207->11510|14281->11561|14344->11595|14461->11684|14475->11688|14548->11739|14611->11773|14714->11848|14728->11852|14796->11898|16095->13168|16125->13169|16176->13191|16248->13234|16278->13235|16443->13371|16473->13372|16524->13394|16597->13438|16627->13439
                  LINES: 20->1|25->1|26->2|48->24|48->24|48->24|73->49|73->49|73->49|75->51|75->51|75->51|76->52|81->57|81->57|81->57|82->58|94->70|94->70|94->70|95->71|95->71|95->71|96->72|96->72|96->72|99->75|99->75|99->75|100->76|119->95|119->95|119->95|123->99|123->99|123->99|124->100|124->100|124->100|124->100|125->101|133->109|133->109|133->109|138->114|138->114|138->114|142->118|142->118|142->118|146->122|146->122|146->122|150->126|150->126|150->126|154->130|154->130|154->130|158->134|158->134|158->134|162->138|162->138|162->138|166->142|166->142|166->142|170->146|170->146|170->146|184->160|184->160|184->160|188->164|188->164|188->164|189->165|190->166|190->166|190->166|191->167|192->168|192->168|192->168|200->176|200->176|200->176|202->178|202->178|202->178|203->179|203->179|203->179|204->180|204->180|204->180|209->185|209->185|209->185|210->186|211->187|211->187|211->187|212->188|213->189|213->189|213->189|237->213|237->213|238->214|239->215|239->215|244->220|244->220|245->221|246->222|246->222
                  -- GENERATED --
              */
          