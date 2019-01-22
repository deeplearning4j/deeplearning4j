
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
            <!-- Start Mobile Specific -->
        <meta name="viewport" content="width=device-width, initial-scale=1">
            <!-- End Mobile Specific -->

        <link id="bootstrap-style" href="/assets/webjars/bootstrap/2.3.1/css/bootstrap.min.css" rel="stylesheet">
        <link id="bootstrap-style" href="/assets/webjars/bootstrap/2.3.1/css/bootstrap-responsive.min.css" rel="stylesheet">
        <link id="base-style" href="/assets/css/style.css" rel="stylesheet">
        <link id="base-style-responsive" href="/assets/css/style-responsive.css" rel="stylesheet">
        <link href='/assets/css/opensans-fonts.css' rel='stylesheet' type='text/css'>
        <link rel="shortcut icon" href="/assets/img/favicon.ico">

            <!-- The HTML5 shim, for IE6-8 support of HTML5 elements -->
            <!--[if lt IE 9]>
	  	<script src="http://html5shim.googlecode.com/svn/trunk/html5.js"></script>
		<link id="ie-style" href="/assets/css/ie.css" rel="stylesheet"/>
	<![endif]-->

            <!--[if IE 9]>
		<link id="ie9style" href="/assets/css/ie9.css" rel="stylesheet"/>
	<![endif]-->

    </head>

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
                    <a class="brand" href="index.html"><span>"""),_display_(/*58.63*/i18n/*58.67*/.getMessage("train.pagetitle")),format.raw/*58.97*/("""</span></a>
                    <div id="sessionSelectDiv" style="display:none; float:right">
                        """),_display_(/*60.26*/i18n/*60.30*/.getMessage("train.session.label")),format.raw/*60.64*/("""
                        """),format.raw/*61.25*/("""<select id="sessionSelect" onchange='selectNewSession()'>
                            <option>(Session ID)</option>
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
                            <li><a href="overview"><i class="icon-bar-chart"></i><span class="hidden-tablet"> """),_display_(/*77.112*/i18n/*77.116*/.getMessage("train.nav.overview")),format.raw/*77.149*/("""</span></a></li>
                            <li><a href="model"><i class="icon-tasks"></i><span class="hidden-tablet"> """),_display_(/*78.105*/i18n/*78.109*/.getMessage("train.nav.model")),format.raw/*78.139*/("""</span></a></li>
                            <li class="active"><a href="javascript:void(0);"><i class="icon-dashboard"></i><span class="hidden-tablet"> """),_display_(/*79.138*/i18n/*79.142*/.getMessage("train.nav.system")),format.raw/*79.173*/("""</span></a></li>
                            """),format.raw/*80.161*/("""
                            """),format.raw/*81.29*/("""<li>
                                <a class="dropmenu" href="javascript:void(0);"><i class="icon-folder-close-alt"></i><span class="hidden-tablet">
                                    """),_display_(/*83.38*/i18n/*83.42*/.getMessage("train.nav.language")),format.raw/*83.75*/("""</span></a>
                                <ul>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('en', 'system')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> English</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('de', 'system')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> Deutsch</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ja', 'system')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> 日本語</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('zh', 'system')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> 中文</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ko', 'system')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> 한글</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ru', 'system')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> русский</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('uk', 'system')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> український</span></a></li>
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
                            JavaScript</a> enabled to use this site.</p>
                    </div>
                </noscript>

                    <!-- Start Content -->
                <div id="content" class="span10">

                    <div class="row-fluid">

                        <div class="box span12">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*114.41*/i18n/*114.45*/.getMessage("train.system.title")),format.raw/*114.78*/("""</b></h2>
                                <div class="btn-group" style="margin-top: -11px; position:absolute; right: 40px;">
                                <button class="btn dropdown-toggle btn-primary" data-toggle="dropdown">"""),_display_(/*116.105*/i18n/*116.109*/.getMessage("train.system.selectMachine")),format.raw/*116.150*/(""" """),format.raw/*116.151*/("""<span class="caret"></span></button>
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
                                                <div class="box-header">
                                                    <h2><b>"""),_display_(/*131.61*/i18n/*131.65*/.getMessage("train.system.chart.systemMemoryTitle")),format.raw/*131.116*/(""" """),format.raw/*131.117*/("""%</b></h2>
                                                </div>
                                                <div class="box-content">
                                                    <div id="systemMemoryChartPlot" class="center" style="height: 300px;" ></div>
                                                    <p id="hoverdata"><b>"""),_display_(/*135.75*/i18n/*135.79*/.getMessage("train.system.chart.memoryShort")),format.raw/*135.124*/(""":</b> <span id="y">0</span>, <b>
                                                        """),_display_(/*136.58*/i18n/*136.62*/.getMessage("train.overview.charts.iteration")),format.raw/*136.108*/(""":</b> <span id="x">0</span></p>
                                                </div>
                                            </div>

                                            <!-- GPU Memory Utlization Chart -->
                                            <div class="box span6" id="gpuMemoryChart">
                                                <div class="box-header">
                                                    <h2><b>"""),_display_(/*143.61*/i18n/*143.65*/.getMessage("train.system.chart.gpuMemoryTitle")),format.raw/*143.113*/(""" """),format.raw/*143.114*/("""%</b></h2>
                                                </div>
                                                <div class="box-content">
                                                    <div id="gpuMemoryChartPlot" class="center" style="height: 300px;" ></div>
                                                    <p id="hoverdata"><b>"""),_display_(/*147.75*/i18n/*147.79*/.getMessage("train.system.chart.memoryShort")),format.raw/*147.124*/(""":</b> <span id="y2">0</span>, <b>
                                                        """),_display_(/*148.58*/i18n/*148.62*/.getMessage("train.overview.charts.iteration")),format.raw/*148.108*/(""":</b> <span id="x2">0</span></p>
                                                </div>
                                            </div>

                                        </div>

                                            <!-- Tables -->
                                        <div class="row-fluid">

                                                <!-- Hardware Information -->
                                            <div class="box span12">
                                                <div class="box-header">
                                                    <h2><b>"""),_display_(/*160.61*/i18n/*160.65*/.getMessage("train.system.hwTable.title")),format.raw/*160.106*/("""</b></h2>
                                                </div>
                                                <div class="box-content">
                                                    <table class="table table-striped">
                                                        <thead>
                                                            <tr>
                                                                <th>"""),_display_(/*166.70*/i18n/*166.74*/.getMessage("train.system.hwTable.jvmCurrent")),format.raw/*166.120*/("""</th>
                                                                <th>"""),_display_(/*167.70*/i18n/*167.74*/.getMessage("train.system.hwTable.jvmMax")),format.raw/*167.116*/("""</th>
                                                                <th>"""),_display_(/*168.70*/i18n/*168.74*/.getMessage("train.system.hwTable.offHeapCurrent")),format.raw/*168.124*/("""</th>
                                                                <th>"""),_display_(/*169.70*/i18n/*169.74*/.getMessage("train.system.hwTable.offHeapMax")),format.raw/*169.120*/("""</th>
                                                                <th>"""),_display_(/*170.70*/i18n/*170.74*/.getMessage("train.system.hwTable.jvmProcs")),format.raw/*170.118*/("""</th>
                                                                <th>"""),_display_(/*171.70*/i18n/*171.74*/.getMessage("train.system.hwTable.computeDevices")),format.raw/*171.124*/("""</th>
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
                                                <div class="box-header">
                                                    <h2><b>"""),_display_(/*195.61*/i18n/*195.65*/.getMessage("train.system.swTable.title")),format.raw/*195.106*/("""</b></h2>
                                                </div>
                                                <div class="box-content">
                                                    <table class="table table-striped">
                                                        <thead>
                                                            <tr>
                                                                <th>"""),_display_(/*201.70*/i18n/*201.74*/.getMessage("train.system.swTable.hostname")),format.raw/*201.118*/("""</th>
                                                                <th>"""),_display_(/*202.70*/i18n/*202.74*/.getMessage("train.system.swTable.os")),format.raw/*202.112*/("""</th>
                                                                <th>"""),_display_(/*203.70*/i18n/*203.74*/.getMessage("train.system.swTable.osArch")),format.raw/*203.116*/("""</th>
                                                                <th>"""),_display_(/*204.70*/i18n/*204.74*/.getMessage("train.system.swTable.jvmName")),format.raw/*204.117*/("""</th>
                                                                <th>"""),_display_(/*205.70*/i18n/*205.74*/.getMessage("train.system.swTable.jvmVersion")),format.raw/*205.120*/("""</th>
                                                                <th>"""),_display_(/*206.70*/i18n/*206.74*/.getMessage("train.system.swTable.nd4jBackend")),format.raw/*206.121*/("""</th>
                                                                <th>"""),_display_(/*207.70*/i18n/*207.74*/.getMessage("train.system.swTable.nd4jDataType")),format.raw/*207.122*/("""</th>
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

                                            """),format.raw/*227.73*/("""
                                        """),format.raw/*228.82*/("""
                                            """),format.raw/*229.73*/("""
                                                """),format.raw/*230.77*/("""
                                                    """),format.raw/*231.88*/("""
                                                """),format.raw/*232.59*/("""
                                                """),format.raw/*233.78*/("""
                                                    """),format.raw/*234.92*/("""
                                                        """),format.raw/*235.68*/("""
                                                            """),format.raw/*236.69*/("""
                                                                """),format.raw/*237.79*/("""
                                                            """),format.raw/*238.70*/("""
                                                        """),format.raw/*239.69*/("""
                                                        """),format.raw/*240.68*/("""
                                                            """),format.raw/*241.69*/("""
                                                                """),format.raw/*242.108*/("""
                                                            """),format.raw/*243.70*/("""
                                                        """),format.raw/*244.69*/("""
                                                    """),format.raw/*245.65*/("""
                                                """),format.raw/*246.59*/("""
                                            """),format.raw/*247.55*/("""
                                        """),format.raw/*248.51*/("""
                                    """),format.raw/*249.37*/("""</div>
                                        <!-- End System Tab -->
                                </div>
                            </div>
                        </div>
                            <!-- End System Tab -->
                    </div><!-- End Row Fluid-->
                </div><!-- End Content -->
            </div><!-- End Container-->
        </div><!-- End Row Fluid-->

        <!-- Start JavaScript-->
        <script src="/assets/webjars/jquery/2.2.0/jquery.min.js"></script>
        <script src="/assets/webjars/jquery-ui/1.10.2/ui/minified/jquery-ui.min.js"></script>
        <script src="/assets/webjars/jquery-migrate/1.2.1/jquery-migrate.min.js"></script>
        <script src="/assets/webjars/jquery-ui-touch-punch/0.2.2/jquery.ui.touch-punch.min.js"></script>
        <script src="/assets/webjars/modernizr/2.8.3/modernizr.min.js"></script>
        <script src="/assets/webjars/bootstrap/2.3.1/js/bootstrap.min.js"></script>
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

        <script src="/assets/js/train/system.js"></script> <!-- Charts and tables are generated here! -->
        <script src="/assets/js/train/train.js"></script>   <!-- Common (lang selection, etc) -->
        <script src="/assets/js/counter.js"></script>

        <!-- Execute once on page load -->
        <script>
                $(document).ready(function () """),format.raw/*291.47*/("""{"""),format.raw/*291.48*/("""
                    """),format.raw/*292.21*/("""renderSystemPage(true);
                    renderTabs();
                    selectMachine();
                    /* Default GPU to hidden */
                    $("#gpuTable").hide();
                    $("#gpuMemoryChart").hide();
                """),format.raw/*298.17*/("""}"""),format.raw/*298.18*/(""");
        </script>

            <!--Execute periodically (every 2 sec) -->
        <script>
                setInterval(function () """),format.raw/*303.41*/("""{"""),format.raw/*303.42*/("""
                    """),format.raw/*304.21*/("""renderSystemPage(false);
                """),format.raw/*305.17*/("""}"""),format.raw/*305.18*/(""", 2000);
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
                  DATE: Tue Jan 22 16:25:54 AEDT 2019
                  SOURCE: c:/DL4J/Git/deeplearning4j/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingSystem.scala.html
                  HASH: 1273170ecfe87a44f26ba407534d696729099794
                  MATRIX: 600->1|733->39|761->41|1679->932|1692->936|1743->966|3473->2669|3486->2673|3537->2703|3685->2824|3698->2828|3753->2862|3807->2888|4539->3592|4553->3596|4608->3629|4758->3751|4772->3755|4824->3785|5007->3940|5021->3944|5074->3975|5149->4153|5207->4183|5423->4372|5436->4376|5490->4409|7925->6816|7939->6820|7994->6853|8254->7084|8269->7088|8333->7129|8364->7130|9225->7963|9239->7967|9313->8018|9344->8019|9720->8367|9734->8371|9802->8416|9921->8507|9935->8511|10004->8557|10480->9005|10494->9009|10565->9057|10596->9058|10969->9403|10983->9407|11051->9452|11171->9544|11185->9548|11254->9594|11887->10199|11901->10203|11965->10244|12424->10675|12438->10679|12507->10725|12611->10801|12625->10805|12690->10847|12794->10923|12808->10927|12881->10977|12985->11053|12999->11057|13068->11103|13172->11179|13186->11183|13253->11227|13357->11303|13371->11307|13444->11357|15091->12976|15105->12980|15169->13021|15628->13452|15642->13456|15709->13500|15813->13576|15827->13580|15888->13618|15992->13694|16006->13698|16071->13740|16175->13816|16189->13820|16255->13863|16359->13939|16373->13943|16442->13989|16546->14065|16560->14069|16630->14116|16734->14192|16748->14196|16819->14244|18219->15643|18290->15726|18365->15800|18444->15878|18527->15967|18606->16027|18685->16106|18768->16199|18855->16268|18946->16338|19041->16418|19132->16489|19219->16559|19306->16628|19397->16698|19493->16807|19584->16878|19671->16948|19754->17014|19833->17074|19908->17130|19979->17182|20046->17220|22873->20018|22903->20019|22954->20041|23240->20298|23270->20299|23438->20438|23468->20439|23519->20461|23590->20503|23620->20504
                  LINES: 20->1|25->1|26->2|48->24|48->24|48->24|82->58|82->58|82->58|84->60|84->60|84->60|85->61|101->77|101->77|101->77|102->78|102->78|102->78|103->79|103->79|103->79|104->80|105->81|107->83|107->83|107->83|138->114|138->114|138->114|140->116|140->116|140->116|140->116|155->131|155->131|155->131|155->131|159->135|159->135|159->135|160->136|160->136|160->136|167->143|167->143|167->143|167->143|171->147|171->147|171->147|172->148|172->148|172->148|184->160|184->160|184->160|190->166|190->166|190->166|191->167|191->167|191->167|192->168|192->168|192->168|193->169|193->169|193->169|194->170|194->170|194->170|195->171|195->171|195->171|219->195|219->195|219->195|225->201|225->201|225->201|226->202|226->202|226->202|227->203|227->203|227->203|228->204|228->204|228->204|229->205|229->205|229->205|230->206|230->206|230->206|231->207|231->207|231->207|251->227|252->228|253->229|254->230|255->231|256->232|257->233|258->234|259->235|260->236|261->237|262->238|263->239|264->240|265->241|266->242|267->243|268->244|269->245|270->246|271->247|272->248|273->249|315->291|315->291|316->292|322->298|322->298|327->303|327->303|328->304|329->305|329->305
                  -- GENERATED --
              */
          