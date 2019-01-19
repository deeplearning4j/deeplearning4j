
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

        <link id="bootstrap-style" href="/assets/css/bootstrap.min.css" rel="stylesheet">
        <link href="/assets/css/bootstrap-responsive.min.css" rel="stylesheet">
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
        <script src="/assets/js/jquery-1.9.1.min.js"></script>
        <script src="/assets/js/jquery-migrate-1.0.0.min.js"></script>
        <script src="/assets/js/jquery-ui-1.10.0.custom.min.js"></script>
        <script src="/assets/js/jquery.ui.touch-punch.js"></script>
        <script src="/assets/js/modernizr.js"></script>
        <script src="/assets/js/bootstrap.min.js"></script>
        <script src="/assets/js/jquery.cookie.js"></script>
        <script src="/assets/js/fullcalendar.min.js"></script>
        <script src="/assets/js/jquery.dataTables.min.js"></script>
        <script src="/assets/js/excanvas.js"></script>
        <script src="/assets/js/jquery.flot.js"></script>
        <script src="/assets/js/jquery.flot.pie.js"></script>
        <script src="/assets/js/jquery.flot.stack.js"></script>
        <script src="/assets/js/jquery.flot.resize.min.js"></script>
        <script src="/assets/js/jquery.chosen.min.js"></script>
        <script src="/assets/js/jquery.uniform.min.js"></script>
        <script src="/assets/js/jquery.cleditor.min.js"></script>
        <script src="/assets/js/jquery.noty.js"></script>
        <script src="/assets/js/jquery.elfinder.min.js"></script>
        <script src="/assets/js/jquery.raty.min.js"></script>
        <script src="/assets/js/jquery.iphone.toggle.js"></script>
        <script src="/assets/js/jquery.uploadify-3.1.min.js"></script>
        <script src="/assets/js/jquery.gritter.min.js"></script>
        <script src="/assets/js/jquery.imagesloaded.js"></script>
        <script src="/assets/js/jquery.masonry.min.js"></script>
        <script src="/assets/js/jquery.knob.modified.js"></script>
        <script src="/assets/js/jquery.sparkline.min.js"></script>
        <script src="/assets/js/counter.js"></script>
        <script src="/assets/js/retina.js"></script>
        <script src="/assets/js/train/system.js"></script> <!-- Charts and tables are generated here! -->
        <script src="/assets/js/train/train.js"></script>   <!-- Common (lang selection, etc) -->

        <!-- Execute once on page load -->
        <script>
                $(document).ready(function () """),format.raw/*295.47*/("""{"""),format.raw/*295.48*/("""
                    """),format.raw/*296.21*/("""renderSystemPage(true);
                    renderTabs();
                    selectMachine();
                    /* Default GPU to hidden */
                    $("#gpuTable").hide();
                    $("#gpuMemoryChart").hide();
                """),format.raw/*302.17*/("""}"""),format.raw/*302.18*/(""");
        </script>

            <!--Execute periodically (every 2 sec) -->
        <script>
                setInterval(function () """),format.raw/*307.41*/("""{"""),format.raw/*307.42*/("""
                    """),format.raw/*308.21*/("""renderSystemPage(false);
                """),format.raw/*309.17*/("""}"""),format.raw/*309.18*/(""", 2000);
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
                  DATE: Sat Jan 19 12:31:33 AEDT 2019
                  SOURCE: c:/DL4J/Git/deeplearning4j/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingSystem.scala.html
                  HASH: 12f99249cd818c138949c0988f653896c553cc10
                  MATRIX: 600->1|733->39|761->41|1679->932|1692->936|1743->966|3404->2600|3417->2604|3468->2634|3616->2755|3629->2759|3684->2793|3738->2819|4470->3523|4484->3527|4539->3560|4689->3682|4703->3686|4755->3716|4938->3871|4952->3875|5005->3906|5080->4084|5138->4114|5354->4303|5367->4307|5421->4340|7856->6747|7870->6751|7925->6784|8185->7015|8200->7019|8264->7060|8295->7061|9156->7894|9170->7898|9244->7949|9275->7950|9651->8298|9665->8302|9733->8347|9852->8438|9866->8442|9935->8488|10411->8936|10425->8940|10496->8988|10527->8989|10900->9334|10914->9338|10982->9383|11102->9475|11116->9479|11185->9525|11818->10130|11832->10134|11896->10175|12355->10606|12369->10610|12438->10656|12542->10732|12556->10736|12621->10778|12725->10854|12739->10858|12812->10908|12916->10984|12930->10988|12999->11034|13103->11110|13117->11114|13184->11158|13288->11234|13302->11238|13375->11288|15022->12907|15036->12911|15100->12952|15559->13383|15573->13387|15640->13431|15744->13507|15758->13511|15819->13549|15923->13625|15937->13629|16002->13671|16106->13747|16120->13751|16186->13794|16290->13870|16304->13874|16373->13920|16477->13996|16491->14000|16561->14047|16665->14123|16679->14127|16750->14175|18150->15574|18221->15657|18296->15731|18375->15809|18458->15898|18537->15958|18616->16037|18699->16130|18786->16199|18877->16269|18972->16349|19063->16420|19150->16490|19237->16559|19328->16629|19424->16738|19515->16809|19602->16879|19685->16945|19764->17005|19839->17061|19910->17113|19977->17151|22639->19784|22669->19785|22720->19807|23006->20064|23036->20065|23204->20204|23234->20205|23285->20227|23356->20269|23386->20270
                  LINES: 20->1|25->1|26->2|48->24|48->24|48->24|82->58|82->58|82->58|84->60|84->60|84->60|85->61|101->77|101->77|101->77|102->78|102->78|102->78|103->79|103->79|103->79|104->80|105->81|107->83|107->83|107->83|138->114|138->114|138->114|140->116|140->116|140->116|140->116|155->131|155->131|155->131|155->131|159->135|159->135|159->135|160->136|160->136|160->136|167->143|167->143|167->143|167->143|171->147|171->147|171->147|172->148|172->148|172->148|184->160|184->160|184->160|190->166|190->166|190->166|191->167|191->167|191->167|192->168|192->168|192->168|193->169|193->169|193->169|194->170|194->170|194->170|195->171|195->171|195->171|219->195|219->195|219->195|225->201|225->201|225->201|226->202|226->202|226->202|227->203|227->203|227->203|228->204|228->204|228->204|229->205|229->205|229->205|230->206|230->206|230->206|231->207|231->207|231->207|251->227|252->228|253->229|254->230|255->231|256->232|257->233|258->234|259->235|260->236|261->237|262->238|263->239|264->240|265->241|266->242|267->243|268->244|269->245|270->246|271->247|272->248|273->249|319->295|319->295|320->296|326->302|326->302|331->307|331->307|332->308|333->309|333->309
                  -- GENERATED --
              */
          