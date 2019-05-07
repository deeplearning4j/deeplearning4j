
package org.deeplearning4j.ui.views.html.training

import play.twirl.api._
import play.twirl.api.TemplateMagic._


     object TrainingSystem_Scope0 {
import models._
import controllers._
import play.api.i18n._
import views.html._
import play.api.templates.PlayMagic._
import play.api.mvc._
import play.api.data._

class TrainingSystem extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template1[org.deeplearning4j.ui.api.I18N,play.twirl.api.HtmlFormat.Appendable] {

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

                    <!-- Start Content -->
                <main id="content" class="main">

                    <div class="row-fluid">

                        <div class="box span12">
                            <div class="chart-header">
                                <h2><b>"""),_display_(/*92.41*/i18n/*92.45*/.getMessage("train.system.title")),format.raw/*92.78*/("""</b></h2>
                                <div class="btn-group" style="margin-top: -11px; position:absolute; right: 40px;">
                                <button class="btn dropdown-toggle btn-primary" data-toggle="dropdown">"""),_display_(/*94.105*/i18n/*94.109*/.getMessage("train.system.selectMachine")),format.raw/*94.150*/(""" """),format.raw/*94.151*/("""<span class="caret"></span></button>
                                    <ul class="dropdown-menu" id="systemTab"></ul>
                                </div>
                            </div>
                            <div class="box-content">

                                    <!--Start System Information -->
                                <div class="tab-content">
                                    <div class="tab-pane active">

                                            <!-- System Memory Utilization Chart -->
                                        <div class="row-fluid">

                                            <div class="box span12" id="systemMemoryChart">
                                                <div class="chart-header">
                                                    <h2><b>"""),_display_(/*109.61*/i18n/*109.65*/.getMessage("train.system.chart.systemMemoryTitle")),format.raw/*109.116*/(""" """),format.raw/*109.117*/("""%</b></h2>
                                                </div>
                                                <div class="box-content">
                                                    <div id="systemMemoryChartPlot" class="center" style="height: 300px;" ></div>
                                                    <p id="hoverdata"><b>"""),_display_(/*113.75*/i18n/*113.79*/.getMessage("train.system.chart.memoryShort")),format.raw/*113.124*/(""":</b> <span id="y">0</span>, <b>
                                                        """),_display_(/*114.58*/i18n/*114.62*/.getMessage("train.overview.charts.iteration")),format.raw/*114.108*/(""":</b> <span id="x">0</span></p>
                                                </div>
                                            </div>

                                            <!-- GPU Memory Utlization Chart -->
                                            <div class="box span6" id="gpuMemoryChart">
                                                <div class="chart-header">
                                                    <h2><b>"""),_display_(/*121.61*/i18n/*121.65*/.getMessage("train.system.chart.gpuMemoryTitle")),format.raw/*121.113*/(""" """),format.raw/*121.114*/("""%</b></h2>
                                                </div>
                                                <div class="box-content">
                                                    <div id="gpuMemoryChartPlot" class="center" style="height: 300px;" ></div>
                                                    <p id="hoverdata"><b>"""),_display_(/*125.75*/i18n/*125.79*/.getMessage("train.system.chart.memoryShort")),format.raw/*125.124*/(""":</b> <span id="y2">0</span>, <b>
                                                        """),_display_(/*126.58*/i18n/*126.62*/.getMessage("train.overview.charts.iteration")),format.raw/*126.108*/(""":</b> <span id="x2">0</span></p>
                                                </div>
                                            </div>

                                        </div>

                                            <!-- Tables -->
                                        <div class="row-fluid">

                                                <!-- Hardware Information -->
                                            <div class="box span12">
                                                <div class="chart-header">
                                                    <h2><b>"""),_display_(/*138.61*/i18n/*138.65*/.getMessage("train.system.hwTable.title")),format.raw/*138.106*/("""</b></h2>
                                                </div>
                                                <div class="box-content">
                                                    <table class="table table-striped">
                                                        <thead>
                                                            <tr>
                                                                <th>"""),_display_(/*144.70*/i18n/*144.74*/.getMessage("train.system.hwTable.jvmCurrent")),format.raw/*144.120*/("""</th>
                                                                <th>"""),_display_(/*145.70*/i18n/*145.74*/.getMessage("train.system.hwTable.jvmMax")),format.raw/*145.116*/("""</th>
                                                                <th>"""),_display_(/*146.70*/i18n/*146.74*/.getMessage("train.system.hwTable.offHeapCurrent")),format.raw/*146.124*/("""</th>
                                                                <th>"""),_display_(/*147.70*/i18n/*147.74*/.getMessage("train.system.hwTable.offHeapMax")),format.raw/*147.120*/("""</th>
                                                                <th>"""),_display_(/*148.70*/i18n/*148.74*/.getMessage("train.system.hwTable.jvmProcs")),format.raw/*148.118*/("""</th>
                                                                <th>"""),_display_(/*149.70*/i18n/*149.74*/.getMessage("train.system.hwTable.computeDevices")),format.raw/*149.124*/("""</th>
                                                            </tr>
                                                        </thead>
                                                        <tbody>
                                                            <tr>
                                                                <td id="currentBytesJVM">Loading...</td>
                                                                <td id="maxBytesJVM">Loading...</td>
                                                                <td id="currentBytesOffHeap">Loading...</td>
                                                                <td id="maxBytesOffHeap">Loading...</td>
                                                                <td id="jvmAvailableProcessors">Loading...</td>
                                                                <td id="nComputeDevices">Loading...</td>
                                                            </tr>
                                                        </tbody>
                                                    </table>
                                                </div>
                                            </div>

                                        </div>

                                        <div class="row-fluid">

                                                <!-- Software Information -->
                                            <div class="box span12">
                                                <div class="chart-header">
                                                    <h2><b>"""),_display_(/*173.61*/i18n/*173.65*/.getMessage("train.system.swTable.title")),format.raw/*173.106*/("""</b></h2>
                                                </div>
                                                <div class="box-content">
                                                    <table class="table table-striped">
                                                        <thead>
                                                            <tr>
                                                                <th>"""),_display_(/*179.70*/i18n/*179.74*/.getMessage("train.system.swTable.hostname")),format.raw/*179.118*/("""</th>
                                                                <th>"""),_display_(/*180.70*/i18n/*180.74*/.getMessage("train.system.swTable.os")),format.raw/*180.112*/("""</th>
                                                                <th>"""),_display_(/*181.70*/i18n/*181.74*/.getMessage("train.system.swTable.osArch")),format.raw/*181.116*/("""</th>
                                                                <th>"""),_display_(/*182.70*/i18n/*182.74*/.getMessage("train.system.swTable.jvmName")),format.raw/*182.117*/("""</th>
                                                                <th>"""),_display_(/*183.70*/i18n/*183.74*/.getMessage("train.system.swTable.jvmVersion")),format.raw/*183.120*/("""</th>
                                                                <th>"""),_display_(/*184.70*/i18n/*184.74*/.getMessage("train.system.swTable.nd4jBackend")),format.raw/*184.121*/("""</th>
                                                                <th>"""),_display_(/*185.70*/i18n/*185.74*/.getMessage("train.system.swTable.nd4jDataType")),format.raw/*185.122*/("""</th>
                                                            </tr>
                                                        </thead>
                                                        <tbody>
                                                            <tr>
                                                                <td id="hostName">Loading...</td>
                                                                <td id="OS">Loading...</td>
                                                                <td id="OSArchitecture">Loading...</td>
                                                                <td id="jvmName">Loading...</td>
                                                                <td id="jvmVersion">Loading...</td>
                                                                <td id="nd4jBackend">Loading...</td>
                                                                <td id="nd4jDataType">Loading...</td>
                                                            </tr>
                                                        </tbody>
                                                    </table>
                                                </div>
                                            </div>

                                        </div>

                                            """),format.raw/*205.73*/("""
                                        """),format.raw/*206.82*/("""
                                            """),format.raw/*207.73*/("""
                                                """),format.raw/*208.79*/("""
                                                    """),format.raw/*209.88*/("""
                                                """),format.raw/*210.59*/("""
                                                """),format.raw/*211.78*/("""
                                                    """),format.raw/*212.92*/("""
                                                        """),format.raw/*213.68*/("""
                                                            """),format.raw/*214.69*/("""
                                                                """),format.raw/*215.79*/("""
                                                            """),format.raw/*216.70*/("""
                                                        """),format.raw/*217.69*/("""
                                                        """),format.raw/*218.68*/("""
                                                            """),format.raw/*219.69*/("""
                                                                """),format.raw/*220.108*/("""
                                                            """),format.raw/*221.70*/("""
                                                        """),format.raw/*222.69*/("""
                                                    """),format.raw/*223.65*/("""
                                                """),format.raw/*224.59*/("""
                                            """),format.raw/*225.55*/("""
                                        """),format.raw/*226.51*/("""
                                    """),format.raw/*227.37*/("""</div>
                                        <!-- End System Tab -->
                                </div>
                            </div>
                        </div>
                            <!-- End System Tab -->
                    </div><!-- End Row Fluid-->
                </main><!-- End Content -->
            </div><!-- End Container-->
        </div><!-- End Row Fluid-->

        <!-- Start JavaScript-->
        <script src="/assets/webjars/jquery/2.2.0/jquery.min.js"></script>
        <script src="/assets/webjars/jquery-ui/1.10.2/ui/minified/jquery-ui.min.js"></script>
        <script src="/assets/webjars/jquery-migrate/1.2.1/jquery-migrate.min.js"></script>
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

        <script src="/assets/js/train/system.js"></script> <!-- Charts and tables are generated here! -->
        <script src="/assets/js/train/train.js"></script>   <!-- Common (lang selection, etc) -->
        <script src="/assets/js/counter.js"></script>

        <!-- Execute once on page load -->
        <script>
                $(document).ready(function () """),format.raw/*262.47*/("""{"""),format.raw/*262.48*/("""
                    """),format.raw/*263.21*/("""renderSystemPage(true);
                    renderTabs();
                    selectMachine();
                    /* Default GPU to hidden */
                    $("#gpuTable").hide();
                    $("#gpuMemoryChart").hide();
                """),format.raw/*269.17*/("""}"""),format.raw/*269.18*/(""");
        </script>

            <!--Execute periodically (every 2 sec) -->
        <script>
                setInterval(function () """),format.raw/*274.41*/("""{"""),format.raw/*274.42*/("""
                    """),format.raw/*275.21*/("""renderSystemPage(false);
                """),format.raw/*276.17*/("""}"""),format.raw/*276.18*/(""", 2000);
        </script>
            <!--End JavaScript-->

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
object TrainingSystem extends TrainingSystem_Scope0.TrainingSystem
              /*
                  -- GENERATED --
                  DATE: Tue May 07 21:39:41 AEST 2019
                  SOURCE: c:/DL4J/Git/deeplearning4j/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingSystem.scala.html
                  HASH: a5fe2fb79cf26f96000f494801fcbb6b9f354dd5
                  MATRIX: 600->1|733->39|761->41|1679->932|1692->936|1743->966|2755->1951|2768->1955|2819->1985|2990->2129|3003->2133|3058->2167|3411->2493|3424->2497|3486->2538|3991->3015|4005->3019|4060->3052|4212->3176|4226->3180|4278->3210|4437->3341|4451->3345|4504->3376|4749->3594|4762->3598|4817->3631|4875->3661|6525->5284|6538->5288|6592->5321|6851->5552|6865->5556|6928->5597|6958->5598|7821->6433|7835->6437|7909->6488|7940->6489|8316->6837|8330->6841|8398->6886|8517->6977|8531->6981|8600->7027|9078->7477|9092->7481|9163->7529|9194->7530|9567->7875|9581->7879|9649->7924|9769->8016|9783->8020|9852->8066|10487->8673|10501->8677|10565->8718|11024->9149|11038->9153|11107->9199|11211->9275|11225->9279|11290->9321|11394->9397|11408->9401|11481->9451|11585->9527|11599->9531|11668->9577|11772->9653|11786->9657|11853->9701|11957->9777|11971->9781|12044->9831|13693->11452|13707->11456|13771->11497|14230->11928|14244->11932|14311->11976|14415->12052|14429->12056|14490->12094|14594->12170|14608->12174|14673->12216|14777->12292|14791->12296|14857->12339|14961->12415|14975->12419|15044->12465|15148->12541|15162->12545|15232->12592|15336->12668|15350->12672|15421->12720|16821->14119|16892->14202|16967->14276|17046->14356|17129->14445|17208->14505|17287->14584|17370->14677|17457->14746|17548->14816|17643->14896|17734->14967|17821->15037|17908->15106|17999->15176|18095->15285|18186->15356|18273->15426|18356->15492|18435->15552|18510->15608|18581->15660|18648->15698|20855->17876|20885->17877|20936->17899|21222->18156|21252->18157|21420->18296|21450->18297|21501->18319|21572->18361|21602->18362
                  LINES: 20->1|25->1|26->2|48->24|48->24|48->24|70->46|70->46|70->46|72->48|72->48|72->48|78->54|78->54|78->54|88->64|88->64|88->64|89->65|89->65|89->65|90->66|90->66|90->66|93->69|93->69|93->69|94->70|116->92|116->92|116->92|118->94|118->94|118->94|118->94|133->109|133->109|133->109|133->109|137->113|137->113|137->113|138->114|138->114|138->114|145->121|145->121|145->121|145->121|149->125|149->125|149->125|150->126|150->126|150->126|162->138|162->138|162->138|168->144|168->144|168->144|169->145|169->145|169->145|170->146|170->146|170->146|171->147|171->147|171->147|172->148|172->148|172->148|173->149|173->149|173->149|197->173|197->173|197->173|203->179|203->179|203->179|204->180|204->180|204->180|205->181|205->181|205->181|206->182|206->182|206->182|207->183|207->183|207->183|208->184|208->184|208->184|209->185|209->185|209->185|229->205|230->206|231->207|232->208|233->209|234->210|235->211|236->212|237->213|238->214|239->215|240->216|241->217|242->218|243->219|244->220|245->221|246->222|247->223|248->224|249->225|250->226|251->227|286->262|286->262|287->263|293->269|293->269|298->274|298->274|299->275|300->276|300->276
                  -- GENERATED --
              */
          