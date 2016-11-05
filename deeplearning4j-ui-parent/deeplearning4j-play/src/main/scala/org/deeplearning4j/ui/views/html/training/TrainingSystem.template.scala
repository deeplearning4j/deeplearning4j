
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
<html lang="en">
    <head>

        <meta charset="utf-8">
        <title>"""),_display_(/*7.17*/i18n/*7.21*/.getMessage("train.pagetitle")),format.raw/*7.51*/("""</title>
            <!-- Start Mobile Specific -->
        <meta name="viewport" content="width=device-width, initial-scale=1">
            <!-- End Mobile Specific -->

        <link id="bootstrap-style" href="/assets/css/bootstrap.min.css" rel="stylesheet">
        <link href="/assets/css/bootstrap-responsive.min.css" rel="stylesheet">
        <link id="base-style" href="/assets/css/style.css" rel="stylesheet">
        <link id="base-style-responsive" href="/assets/css/style-responsive.css" rel="stylesheet">
        <link href='http://fonts.googleapis.com/css?family=Open+Sans:300italic,400italic,600italic,700italic,800italic,400,300,600,700,800&subset=latin,cyrillic-ext,latin-ext' rel='stylesheet' type='text/css'>
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
                    <a class="brand" href="index.html"><span>"""),_display_(/*41.63*/i18n/*41.67*/.getMessage("train.pagetitle")),format.raw/*41.97*/("""</span></a>
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
                            <li><a href="overview"><i class="icon-bar-chart"></i><span class="hidden-tablet"> """),_display_(/*54.112*/i18n/*54.116*/.getMessage("train.nav.overview")),format.raw/*54.149*/("""</span></a></li>
                            <li><a href="model"><i class="icon-tasks"></i><span class="hidden-tablet"> """),_display_(/*55.105*/i18n/*55.109*/.getMessage("train.nav.model")),format.raw/*55.139*/("""</span></a></li>
                            <li class="active"><a href="javascript:void(0);"><i class="icon-dashboard"></i><span class="hidden-tablet"> """),_display_(/*56.138*/i18n/*56.142*/.getMessage("train.nav.system")),format.raw/*56.173*/("""</span></a></li>
                            <li><a href="help"><i class="icon-star"></i><span class="hidden-tablet"> """),_display_(/*57.103*/i18n/*57.107*/.getMessage("train.nav.userguide")),format.raw/*57.141*/("""</span></a></li>
                            <li>
                                <a class="dropmenu" href="javascript:void(0);"><i class="icon-folder-close-alt"></i><span class="hidden-tablet">
                                    Language</span></a>
                                <ul>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('en', 'system')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> English</span></a></li>
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
                                <h2><b>"""),_display_(/*90.41*/i18n/*90.45*/.getMessage("train.system.title")),format.raw/*90.78*/("""</b></h2>
                            </div>
                            <div class="box-content">
                                <ul class="nav tab-menu nav-tabs" id="systemTab"></ul>

                                    <!--Start System Tab -->
                                <div id="myTabContent" class="tab-content">
                                    <div class="tab-pane active">

                                            <!-- System Memory Utilization Chart -->
                                        <div class="row-fluid">

                                            <div class="box span12" id="systemMemoryChart">
                                                <div class="box-header">
                                                    <h2><b>"""),_display_(/*104.61*/i18n/*104.65*/.getMessage("train.system.chart.systemTitle")),format.raw/*104.110*/(""" """),format.raw/*104.111*/("""%</b></h2>
                                                </div>
                                                <div class="box-content">
                                                    <div id="systemMemoryChartPlot" class="center" style="height: 300px;" ></div>
                                                    <p id="hoverdata"><b>"""),_display_(/*108.75*/i18n/*108.79*/.getMessage("train.system.chart.memoryShort")),format.raw/*108.124*/(""":</b> <span id="y">0</span>, <b>
                                                        """),_display_(/*109.58*/i18n/*109.62*/.getMessage("train.overview.charts.iteration")),format.raw/*109.108*/(""":</b> <span id="x">0</span></p>
                                                </div>
                                            </div>

                                            <!-- GPU Memory Utlization Chart -->
                                            <div class="box span6" id="gpuMemoryChart">
                                                <div class="box-header">
                                                    <h2><b>"""),_display_(/*116.61*/i18n/*116.65*/.getMessage("train.system.chart.gpuTitle")),format.raw/*116.107*/(""" """),format.raw/*116.108*/("""%</b></h2>
                                                </div>
                                                <div class="box-content">
                                                    <div id="gpuMemoryChartPlot" class="center" style="height: 300px;" ></div>
                                                    <p id="hoverdata"><b>"""),_display_(/*120.75*/i18n/*120.79*/.getMessage("train.system.chart.memoryShort")),format.raw/*120.124*/(""":</b> <span id="y2">0</span>, <b>
                                                        """),_display_(/*121.58*/i18n/*121.62*/.getMessage("train.overview.charts.iteration")),format.raw/*121.108*/(""":</b> <span id="x2">0</span></p>
                                                </div>
                                            </div>

                                        </div>

                                            <!-- Tables -->
                                        <div class="row-fluid">

                                                <!-- Hardware Information -->
                                            <div class="box span12">
                                                <div class="box-header">
                                                    <h2><b>"""),_display_(/*133.61*/i18n/*133.65*/.getMessage("train.system.hwTable.title")),format.raw/*133.106*/("""</b></h2>
                                                </div>
                                                <div class="box-content">
                                                    <table class="table table-striped">
                                                        <thead>
                                                            <tr>
                                                                <th>"""),_display_(/*139.70*/i18n/*139.74*/.getMessage("train.system.hwTable.jvmCurrent")),format.raw/*139.120*/("""</th>
                                                                <th>"""),_display_(/*140.70*/i18n/*140.74*/.getMessage("train.system.hwTable.jvmMax")),format.raw/*140.116*/("""</th>
                                                                <th>"""),_display_(/*141.70*/i18n/*141.74*/.getMessage("train.system.hwTable.offHeapCurrent")),format.raw/*141.124*/("""</th>
                                                                <th>"""),_display_(/*142.70*/i18n/*142.74*/.getMessage("train.system.hwTable.offHeapMax")),format.raw/*142.120*/("""</th>
                                                                <th>"""),_display_(/*143.70*/i18n/*143.74*/.getMessage("train.system.hwTable.jvmProcs")),format.raw/*143.118*/("""</th>
                                                                <th>"""),_display_(/*144.70*/i18n/*144.74*/.getMessage("train.system.hwTable.computeDevices")),format.raw/*144.124*/("""</th>
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
                                                    <h2><b>"""),_display_(/*168.61*/i18n/*168.65*/.getMessage("train.system.swTable.title")),format.raw/*168.106*/("""</b></h2>
                                                </div>
                                                <div class="box-content">
                                                    <table class="table table-striped">
                                                        <thead>
                                                            <tr>
                                                                <th>"""),_display_(/*174.70*/i18n/*174.74*/.getMessage("train.system.swTable.hostname")),format.raw/*174.118*/("""</th>
                                                                <th>"""),_display_(/*175.70*/i18n/*175.74*/.getMessage("train.system.swTable.os")),format.raw/*175.112*/("""</th>
                                                                <th>"""),_display_(/*176.70*/i18n/*176.74*/.getMessage("train.system.swTable.osArch")),format.raw/*176.116*/("""</th>
                                                                <th>"""),_display_(/*177.70*/i18n/*177.74*/.getMessage("train.system.swTable.jvmName")),format.raw/*177.117*/("""</th>
                                                                <th>"""),_display_(/*178.70*/i18n/*178.74*/.getMessage("train.system.swTable.jvmVersion")),format.raw/*178.120*/("""</th>
                                                                <th>"""),_display_(/*179.70*/i18n/*179.74*/.getMessage("train.system.swTable.nd4jBackend")),format.raw/*179.121*/("""</th>
                                                                <th>"""),_display_(/*180.70*/i18n/*180.74*/.getMessage("train.system.swTable.nd4jDataType")),format.raw/*180.122*/("""</th>
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

                                            <!-- GPU Information -->
                                        <div class="row-fluid" id="gpuTable">
                                            <div class="box span12">
                                                <div class="box-header">
                                                    <h2><b>GPU Information</b></h2>
                                                </div>
                                                <div class="box-content">
                                                    <table class="table table-striped">
                                                        <thead>
                                                            <tr>
                                                                <th>?</th>
                                                                <th>?</th>
                                                                <th>?</th>
                                                                <th>?</th>
                                                                <th>?</th>
                                                                <th>?</th>
                                                                <th>?</th>
                                                            </tr>
                                                        </thead>
                                                        <tbody>
                                                            <tr>
                                                                <td id="gpuPlaceholder">Loading...</td>
                                                                <td id="gpuPlaceholder">Loading...</td>
                                                                <td id="gpuPlaceholder">Loading...</td>
                                                                <td id="gpuPlaceholder">Loading...</td>
                                                                <td id="gpuPlaceholder">Loading...</td>
                                                                <td id="gpuPlaceholder">Loading...</td>
                                                                <td id="gpuPlaceholder">Loading...</td>
                                                            </tr>
                                                        </tbody>
                                                    </table>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
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
                $(document).ready(function () """),format.raw/*280.47*/("""{"""),format.raw/*280.48*/("""
                    """),format.raw/*281.21*/("""renderSystemPage();
                    renderTabs();
                    selectMachine();
                """),format.raw/*284.17*/("""}"""),format.raw/*284.18*/(""");
        </script>

            <!--Execute periodically (every 2 sec) -->
        <script>
                setInterval(function () """),format.raw/*289.41*/("""{"""),format.raw/*289.42*/("""
                    """),format.raw/*290.21*/("""renderSystemPage();
                """),format.raw/*291.17*/("""}"""),format.raw/*291.18*/(""", 2000);
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
                  DATE: Sat Nov 05 23:29:02 AEDT 2016
                  SOURCE: C:/DL4J/Git/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingSystem.scala.html
                  HASH: e342d11537d69f80b3bf16ff26fdc6c36dbc8f3e
                  MATRIX: 600->1|733->39|761->41|884->138|896->142|946->172|2731->1930|2744->1934|2795->1964|3359->2500|3373->2504|3428->2537|3578->2659|3592->2663|3644->2693|3827->2848|3841->2852|3894->2883|4042->3003|4056->3007|4112->3041|6577->5479|6590->5483|6644->5516|7452->6296|7466->6300|7534->6345|7565->6346|7941->6694|7955->6698|8023->6743|8142->6834|8156->6838|8225->6884|8701->7332|8715->7336|8780->7378|8811->7379|9184->7724|9198->7728|9266->7773|9386->7865|9400->7869|9469->7915|10102->8520|10116->8524|10180->8565|10639->8996|10653->9000|10722->9046|10826->9122|10840->9126|10905->9168|11009->9244|11023->9248|11096->9298|11200->9374|11214->9378|11283->9424|11387->9500|11401->9504|11468->9548|11572->9624|11586->9628|11659->9678|13306->11297|13320->11301|13384->11342|13843->11773|13857->11777|13924->11821|14028->11897|14042->11901|14103->11939|14207->12015|14221->12019|14286->12061|14390->12137|14404->12141|14470->12184|14574->12260|14588->12264|14657->12310|14761->12386|14775->12390|14845->12437|14949->12513|14963->12517|15034->12565|21670->19172|21700->19173|21751->19195|21890->19305|21920->19306|22088->19445|22118->19446|22169->19468|22235->19505|22265->19506
                  LINES: 20->1|25->1|26->2|31->7|31->7|31->7|65->41|65->41|65->41|78->54|78->54|78->54|79->55|79->55|79->55|80->56|80->56|80->56|81->57|81->57|81->57|114->90|114->90|114->90|128->104|128->104|128->104|128->104|132->108|132->108|132->108|133->109|133->109|133->109|140->116|140->116|140->116|140->116|144->120|144->120|144->120|145->121|145->121|145->121|157->133|157->133|157->133|163->139|163->139|163->139|164->140|164->140|164->140|165->141|165->141|165->141|166->142|166->142|166->142|167->143|167->143|167->143|168->144|168->144|168->144|192->168|192->168|192->168|198->174|198->174|198->174|199->175|199->175|199->175|200->176|200->176|200->176|201->177|201->177|201->177|202->178|202->178|202->178|203->179|203->179|203->179|204->180|204->180|204->180|304->280|304->280|305->281|308->284|308->284|313->289|313->289|314->290|315->291|315->291
                  -- GENERATED --
              */
          