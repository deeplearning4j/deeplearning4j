
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

        <!-- Icons -->
        <link rel="stylesheet" href="/assets/webjars/coreui__icons/0.3.0/css/coreui-icons.min.css"></script>


        <link rel="shortcut icon" href="/assets/img/favicon.ico">
    </head>

    <body class="app sidebar-show aside-menu-show">
        <header class="app-header navbar">
                <a class="header-text" href="#"><span>"""),_display_(/*45.56*/i18n/*45.60*/.getMessage("train.pagetitle")),format.raw/*45.90*/("""</span></a>
                <div id="sessionSelectDiv" style="display:none; float:right;">
                        <div style="color:white;">"""),_display_(/*47.52*/i18n/*47.56*/.getMessage("train.session.label")),format.raw/*47.90*/("""</div>
                        <select id="sessionSelect" onchange='selectNewSession()'>
                        <option>(Session ID)</option>
                </select>
                </div>
                <div id="workerSelectDiv" style="display:none; float:right">
                        <div style="color:white;">"""),_display_(/*53.52*/i18n/*53.56*/.getMessage("train.session.worker.label")),format.raw/*53.97*/("""</div>
                        <select id="workerSelect" onchange='selectNewWorker()'>
                        <option>(Worker ID)</option>
                </select>
                </div>
        </header>
        <div class="app-body">
            <div class="sidebar">
                <nav class="sidebar-nav">
                    <ul class="nav">
                        <li class="nav-item"><a class="nav-link" href="overview"><i class="nav-icon cui-chart"></i>"""),_display_(/*63.117*/i18n/*63.121*/.getMessage("train.nav.overview")),format.raw/*63.154*/("""</a></li>
                        <li class="nav-item"><a class="nav-link" href="model"><i class="nav-icon cui-graph"></i>"""),_display_(/*64.114*/i18n/*64.118*/.getMessage("train.nav.model")),format.raw/*64.148*/("""</a></li>
                        <li class="nav-item"><a class="nav-link" href="system"><i class="nav-icon cui-speedometer"></i>"""),_display_(/*65.121*/i18n/*65.125*/.getMessage("train.nav.system")),format.raw/*65.156*/("""</a></li>
                        <li class="nav-item nav-dropdown">
                            <a class="nav-link nav-dropdown-toggle" href="#">
                                <i class="nav-icon cui-globe"></i> """),_display_(/*68.69*/i18n/*68.73*/.getMessage("train.nav.language")),format.raw/*68.106*/("""
                            """),format.raw/*69.29*/("""</a>
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
                            <h2><b>"""),_display_(/*88.37*/i18n/*88.41*/.getMessage("train.overview.chart.scoreTitle")),format.raw/*88.87*/("""</b></h2>
                        </div>
                        <div>
                            <div id="scoreiterchart" class="center" style="height: 300px;" ></div>
                            <p id="hoverdata"><b>"""),_display_(/*92.51*/i18n/*92.55*/.getMessage("train.overview.chart.scoreTitleShort")),format.raw/*92.106*/("""
                                """),format.raw/*93.33*/(""":</b> <span id="y">0</span>, <b>"""),_display_(/*93.66*/i18n/*93.70*/.getMessage("train.overview.charts.iteration")),format.raw/*93.116*/("""
                                """),format.raw/*94.33*/(""":</b> <span id="x">
                                0</span></p>
                        </div>
                    </div>
                        <!-- End Score Chart-->
                        <!-- Start Model Table-->
                    <div class="col-4 chart-box">
                        <div class="chart-header">
                            <h2><b>"""),_display_(/*102.37*/i18n/*102.41*/.getMessage("train.overview.perftable.title")),format.raw/*102.86*/("""</b></h2>
                        </div>
                        <div>
                            <table class="table table-bordered table-striped table-condensed">
                                <tr>
                                    <td>"""),_display_(/*107.42*/i18n/*107.46*/.getMessage("train.overview.modeltable.modeltype")),format.raw/*107.96*/("""</td>
                                    <td id="modelType">Loading...</td>
                                </tr>
                                <tr>
                                    <td>"""),_display_(/*111.42*/i18n/*111.46*/.getMessage("train.overview.modeltable.nLayers")),format.raw/*111.94*/("""</td>
                                    <td id="nLayers">Loading...</td>
                                </tr>
                                <tr>
                                    <td>"""),_display_(/*115.42*/i18n/*115.46*/.getMessage("train.overview.modeltable.nParams")),format.raw/*115.94*/("""</td>
                                    <td id="nParams">Loading...</td>
                                </tr>
                                <tr>
                                    <td>"""),_display_(/*119.42*/i18n/*119.46*/.getMessage("train.overview.perftable.startTime")),format.raw/*119.95*/("""</td>
                                    <td id="startTime">Loading...</td>
                                </tr>
                                <tr>
                                    <td>"""),_display_(/*123.42*/i18n/*123.46*/.getMessage("train.overview.perftable.totalRuntime")),format.raw/*123.98*/("""</td>
                                    <td id="totalRuntime">Loading...</td>
                                </tr>
                                <tr>
                                    <td>"""),_display_(/*127.42*/i18n/*127.46*/.getMessage("train.overview.perftable.lastUpdate")),format.raw/*127.96*/("""</td>
                                    <td id="lastUpdate">Loading...</td>
                                </tr>
                                <tr>
                                    <td>"""),_display_(/*131.42*/i18n/*131.46*/.getMessage("train.overview.perftable.totalParamUpdates")),format.raw/*131.103*/("""</td>
                                    <td id="totalParamUpdates">Loading...</td>
                                </tr>
                                <tr>
                                    <td>"""),_display_(/*135.42*/i18n/*135.46*/.getMessage("train.overview.perftable.updatesPerSec")),format.raw/*135.99*/("""</td>
                                    <td id="updatesPerSec">Loading...</td>
                                </tr>
                                <tr>
                                    <td>"""),_display_(/*139.42*/i18n/*139.46*/.getMessage("train.overview.perftable.examplesPerSec")),format.raw/*139.100*/("""</td>
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
                            <h2><b>"""),_display_(/*153.37*/i18n/*153.41*/.getMessage("train.overview.chart.updateRatioTitle")),format.raw/*153.93*/(""": log<sub>10</sub></b></h2>
                        </div>
                        <div class="box-content">
                            <div id="updateRatioChart" class="center" style="height: 300px;" ></div>
                            <p id="hoverdata"><b>"""),_display_(/*157.51*/i18n/*157.55*/.getMessage("train.overview.chart.updateRatioTitleShort")),format.raw/*157.112*/("""
                                """),format.raw/*158.33*/(""":</b> <span id="yRatio">0</span>, <b>log<sub>
                                10</sub> """),_display_(/*159.43*/i18n/*159.47*/.getMessage("train.overview.chart.updateRatioTitleShort")),format.raw/*159.104*/("""
                                """),format.raw/*160.33*/(""":</b> <span id="yLogRatio">0</span>
                                , <b>"""),_display_(/*161.39*/i18n/*161.43*/.getMessage("train.overview.charts.iteration")),format.raw/*161.89*/(""":</b> <span id="xRatio">
                                    0</span></p>
                        </div>

                    </div>
                        <!--End Ratio Table -->
                        <!--Start Variance Table -->
                    <div class="col chart-box">
                        <div class="chart-header">
                            <h2><b>"""),_display_(/*170.37*/i18n/*170.41*/.getMessage("train.overview.chart.stdevTitle")),format.raw/*170.87*/(""": log<sub>10</sub></b></h2>
                            <ul class="nav tab-menu nav-tabs" style="position:absolute; margin-top: -30px; right: 22px;">
                                <li class="active" id="stdevActivations"><a href="javascript:void(0);" onclick="selectStdevChart('stdevActivations')">"""),_display_(/*172.152*/i18n/*172.156*/.getMessage("train.overview.chart.stdevBtn.activations")),format.raw/*172.212*/("""</a></li>
                                <li id="stdevGradients"><a href="javascript:void(0);" onclick="selectStdevChart('stdevGradients')">"""),_display_(/*173.133*/i18n/*173.137*/.getMessage("train.overview.chart.stdevBtn.gradients")),format.raw/*173.191*/("""</a></li>
                                <li id="stdevUpdates"><a href="javascript:void(0);" onclick="selectStdevChart('stdevUpdates')">"""),_display_(/*174.129*/i18n/*174.133*/.getMessage("train.overview.chart.stdevBtn.updates")),format.raw/*174.185*/("""</a></li>
                            </ul>
                        </div>
                        <div class="box-content">
                            <div id="stdevChart" class="center" style="height: 300px;" ></div>
                            <p id="hoverdata"><b>"""),_display_(/*179.51*/i18n/*179.55*/.getMessage("train.overview.chart.stdevTitleShort")),format.raw/*179.106*/("""
                                """),format.raw/*180.33*/(""":</b> <span id="yStdev">0</span>, <b>log<sub>
                                10</sub> """),_display_(/*181.43*/i18n/*181.47*/.getMessage("train.overview.chart.stdevTitleShort")),format.raw/*181.98*/("""
                                """),format.raw/*182.33*/(""":</b> <span id="yLogStdev">0</span>
                                , <b>"""),_display_(/*183.39*/i18n/*183.43*/.getMessage("train.overview.charts.iteration")),format.raw/*183.89*/(""":</b> <span id="xStdev">
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
                $(document).ready(function () """),format.raw/*207.47*/("""{"""),format.raw/*207.48*/("""
                    """),format.raw/*208.21*/("""renderOverviewPage(true);
                """),format.raw/*209.17*/("""}"""),format.raw/*209.18*/(""");
        </script>

        <!-- Execute periodically (every 2 sec) -->
        <script>
                setInterval(function () """),format.raw/*214.41*/("""{"""),format.raw/*214.42*/("""
                    """),format.raw/*215.21*/("""renderOverviewPage(false);
                """),format.raw/*216.17*/("""}"""),format.raw/*216.18*/(""", 2000);
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
                  DATE: Tue May 07 18:12:43 AEST 2019
                  SOURCE: c:/DL4J/Git/deeplearning4j/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingOverview.scala.html
                  HASH: f8d4eb45ba11028c15676e8b2f9d4c3219612392
                  MATRIX: 604->1|737->39|765->41|1683->932|1696->936|1747->966|2787->1979|2800->1983|2851->2013|3022->2157|3035->2161|3090->2195|3443->2521|3456->2525|3518->2566|4023->3043|4037->3047|4092->3080|4244->3204|4258->3208|4310->3238|4469->3369|4483->3373|4536->3404|4781->3622|4794->3626|4849->3659|4907->3689|6488->5243|6501->5247|6568->5293|6819->5517|6832->5521|6905->5572|6967->5606|7027->5639|7040->5643|7108->5689|7170->5723|7564->6089|7578->6093|7645->6138|7922->6387|7936->6391|8008->6441|8233->6638|8247->6642|8317->6690|8540->6885|8554->6889|8624->6937|8847->7132|8861->7136|8932->7185|9157->7382|9171->7386|9245->7438|9473->7638|9487->7642|9559->7692|9785->7890|9799->7894|9879->7951|10112->8156|10126->8160|10201->8213|10430->8414|10444->8418|10521->8472|11070->8993|11084->8997|11158->9049|11450->9313|11464->9317|11544->9374|11607->9408|11724->9497|11738->9501|11818->9558|11881->9592|11984->9667|11998->9671|12066->9717|12472->10095|12486->10099|12554->10145|12886->10448|12901->10452|12980->10508|13152->10651|13167->10655|13244->10709|13412->10848|13427->10852|13502->10904|13805->11179|13819->11183|13893->11234|13956->11268|14073->11357|14087->11361|14160->11412|14223->11446|14326->11521|14340->11525|14408->11571|15707->12841|15737->12842|15788->12864|15860->12907|15890->12908|16055->13044|16085->13045|16136->13067|16209->13111|16239->13112
                  LINES: 20->1|25->1|26->2|48->24|48->24|48->24|69->45|69->45|69->45|71->47|71->47|71->47|77->53|77->53|77->53|87->63|87->63|87->63|88->64|88->64|88->64|89->65|89->65|89->65|92->68|92->68|92->68|93->69|112->88|112->88|112->88|116->92|116->92|116->92|117->93|117->93|117->93|117->93|118->94|126->102|126->102|126->102|131->107|131->107|131->107|135->111|135->111|135->111|139->115|139->115|139->115|143->119|143->119|143->119|147->123|147->123|147->123|151->127|151->127|151->127|155->131|155->131|155->131|159->135|159->135|159->135|163->139|163->139|163->139|177->153|177->153|177->153|181->157|181->157|181->157|182->158|183->159|183->159|183->159|184->160|185->161|185->161|185->161|194->170|194->170|194->170|196->172|196->172|196->172|197->173|197->173|197->173|198->174|198->174|198->174|203->179|203->179|203->179|204->180|205->181|205->181|205->181|206->182|207->183|207->183|207->183|231->207|231->207|232->208|233->209|233->209|238->214|238->214|239->215|240->216|240->216
                  -- GENERATED --
              */
          