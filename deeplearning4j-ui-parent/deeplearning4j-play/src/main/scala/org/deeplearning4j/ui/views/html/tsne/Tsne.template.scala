
package org.deeplearning4j.ui.views.html.tsne

import play.twirl.api._
import play.twirl.api.TemplateMagic._


     object Tsne_Scope0 {
import models._
import controllers._
import play.api.i18n._
import views.html._
import play.api.templates.PlayMagic._
import play.api.mvc._
import play.api.data._

class Tsne extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template0[play.twirl.api.HtmlFormat.Appendable] {

  /**/
  def apply():play.twirl.api.HtmlFormat.Appendable = {
    _display_ {
      {


Seq[Any](format.raw/*1.1*/("""<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="utf-8" />
        <title>T-SNE renders</title>
            <!-- jQuery -->
        <script src="/assets/legacy/jquery-2.2.0.min.js"></script>

        <link href='/assets/legacy/roboto.css' rel='stylesheet' type='text/css'>

            <!-- Latest compiled and minified CSS -->
        <link rel="stylesheet" href="/assets/legacy/bootstrap.min.css" />

            <!-- Optional theme -->
        <link rel="stylesheet" href="/assets/legacy/bootstrap-theme.min.css" />

            <!-- Latest compiled and minified JavaScript -->
        <script src="/assets/legacy/bootstrap.min.js" ></script>


            <!-- d3 -->
        <script src="/assets/legacy/d3.v3.min.js" charset="utf-8"></script>


        <script src="/assets/legacy/jquery-fileupload.js"></script>

            <!-- Booststrap Notify plugin-->
        <script src="/assets/legacy/bootstrap-notify.min.js"></script>

        <script src="/assets/legacy/common.js"></script>

            <!-- dl4j plot setup -->
        <script src="/assets/legacy/renderTsne.js"></script>


        <style>
        .hd """),format.raw/*37.13*/("""{"""),format.raw/*37.14*/("""
            """),format.raw/*38.13*/("""background-color: #000000;
            font-size: 18px;
            color: #FFFFFF;
        """),format.raw/*41.9*/("""}"""),format.raw/*41.10*/("""

        """),format.raw/*43.9*/(""".block """),format.raw/*43.16*/("""{"""),format.raw/*43.17*/("""
            """),format.raw/*44.13*/("""width: 250px;
            height: 350px;
            display: inline-block;
            border: 1px solid #DEDEDE;
            margin-right: 64px;
        """),format.raw/*49.9*/("""}"""),format.raw/*49.10*/("""

        """),format.raw/*51.9*/(""".hd-small """),format.raw/*51.19*/("""{"""),format.raw/*51.20*/("""
            """),format.raw/*52.13*/("""background-color: #000000;
            font-size: 14px;
            color: #FFFFFF;
        """),format.raw/*55.9*/("""}"""),format.raw/*55.10*/("""

        """),format.raw/*57.9*/("""body """),format.raw/*57.14*/("""{"""),format.raw/*57.15*/("""
            """),format.raw/*58.13*/("""font-family: 'Roboto', sans-serif;
            color: #333;
            font-weight: 300;
            font-size: 16px;
        """),format.raw/*62.9*/("""}"""),format.raw/*62.10*/("""

        """),format.raw/*64.9*/("""#wrap """),format.raw/*64.15*/("""{"""),format.raw/*64.16*/("""
            """),format.raw/*65.13*/("""width: 800px;
            margin-left: auto;
            margin-right: auto;
        """),format.raw/*68.9*/("""}"""),format.raw/*68.10*/("""

        """),format.raw/*70.9*/("""#embed """),format.raw/*70.16*/("""{"""),format.raw/*70.17*/("""
            """),format.raw/*71.13*/("""margin-top: 10px;
        """),format.raw/*72.9*/("""}"""),format.raw/*72.10*/("""

        """),format.raw/*74.9*/("""h1 """),format.raw/*74.12*/("""{"""),format.raw/*74.13*/("""
            """),format.raw/*75.13*/("""text-align: center;
            font-weight: normal;
        """),format.raw/*77.9*/("""}"""),format.raw/*77.10*/("""

        """),format.raw/*79.9*/(""".tt """),format.raw/*79.13*/("""{"""),format.raw/*79.14*/("""
            """),format.raw/*80.13*/("""margin-top: 10px;
            background-color: #EEE;
            border-bottom: 1px solid #333;
            padding: 5px;
        """),format.raw/*84.9*/("""}"""),format.raw/*84.10*/("""

        """),format.raw/*86.9*/(""".txth """),format.raw/*86.15*/("""{"""),format.raw/*86.16*/("""
            """),format.raw/*87.13*/("""color: #F55;
        """),format.raw/*88.9*/("""}"""),format.raw/*88.10*/("""

        """),format.raw/*90.9*/(""".cit """),format.raw/*90.14*/("""{"""),format.raw/*90.15*/("""
            """),format.raw/*91.13*/("""font-family: courier;
            padding-left: 20px;
            font-size: 14px;
        """),format.raw/*94.9*/("""}"""),format.raw/*94.10*/("""
        """),format.raw/*95.9*/("""</style>

        <script>
        $(document).ready(function () """),format.raw/*98.39*/("""{"""),format.raw/*98.40*/("""
            """),format.raw/*99.13*/("""$('#filenamebutton').click(function () """),format.raw/*99.52*/("""{"""),format.raw/*99.53*/("""
                """),format.raw/*100.17*/("""document.getElementById('form').reset();
                $('#form').hide();
                var filename = $('#filename').val();
                $('#filename').val('');
                updateFileName(filename);
                drawTsne();
            """),format.raw/*106.13*/("""}"""),format.raw/*106.14*/(""");

            $('#form').fileUpload("""),format.raw/*108.35*/("""{"""),format.raw/*108.36*/("""
                """),format.raw/*109.17*/("""success: function (data, textStatus, jqXHR) """),format.raw/*109.61*/("""{"""),format.raw/*109.62*/("""
                    """),format.raw/*110.21*/("""var fullPath = document.getElementById('form').value;
                    var filename = data['name'];
                    if (fullPath) """),format.raw/*112.35*/("""{"""),format.raw/*112.36*/("""
                        """),format.raw/*113.25*/("""var startIndex = (fullPath.indexOf('\\') >= 0 ? fullPath.lastIndexOf('\\') : fullPath.lastIndexOf('/'));
                        var filename = fullPath.substring(startIndex);
                        if (filename.indexOf('\\') === 0 || filename.indexOf('/') === 0) """),format.raw/*115.90*/("""{"""),format.raw/*115.91*/("""
                            """),format.raw/*116.29*/("""filename = filename.substring(1);
                        """),format.raw/*117.25*/("""}"""),format.raw/*117.26*/("""
                    """),format.raw/*118.21*/("""}"""),format.raw/*118.22*/("""

                    """),format.raw/*120.21*/("""document.getElementById('form').reset();

                    updateFileName(filename);
                    drawTsne();
                """),format.raw/*124.17*/("""}"""),format.raw/*124.18*/(""", error: function (err) """),format.raw/*124.42*/("""{"""),format.raw/*124.43*/("""
                    """),format.raw/*125.21*/("""console.log(err);
                    drawTsne();
                """),format.raw/*127.17*/("""}"""),format.raw/*127.18*/("""
            """),format.raw/*128.13*/("""}"""),format.raw/*128.14*/(""");


            function updateFileName(name) """),format.raw/*131.43*/("""{"""),format.raw/*131.44*/("""
                """),format.raw/*132.17*/("""$.ajax("""),format.raw/*132.24*/("""{"""),format.raw/*132.25*/("""
                    """),format.raw/*133.21*/("""url: '/tsne/upload',
                    type: 'POST',
                    dataType: 'json',
                    data: JSON.stringify("""),format.raw/*136.42*/("""{"""),format.raw/*136.43*/(""""url": name"""),format.raw/*136.54*/("""}"""),format.raw/*136.55*/("""),
                    cache: false,
                    success: function (data, textStatus, jqXHR) """),format.raw/*138.65*/("""{"""),format.raw/*138.66*/("""
                        """),format.raw/*139.25*/("""setSessionId("UploadedFile");
                        drawTsne();
                    """),format.raw/*141.21*/("""}"""),format.raw/*141.22*/(""",
                    error: function (jqXHR, textStatus, errorThrown) """),format.raw/*142.70*/("""{"""),format.raw/*142.71*/("""
                        """),format.raw/*143.25*/("""// Handle errors here
                        console.log('ERRORS: ' + textStatus);
                        drawTsne();
                    """),format.raw/*146.21*/("""}"""),format.raw/*146.22*/("""
                """),format.raw/*147.17*/("""}"""),format.raw/*147.18*/(""");
            """),format.raw/*148.13*/("""}"""),format.raw/*148.14*/("""


        """),format.raw/*151.9*/("""}"""),format.raw/*151.10*/(""") ;

    </script>

    </head>

    <body>
        <table style="width: 100%;
            padding: 5px;" class="hd">
            <tbody>
                <tr>
                    <td style="width: 48px;"><a href="/"><img src="/assets/legacy/deeplearning4j.img" border="0"/></a></td>
                    <td>DeepLearning4j UI</td>
                    <td style="width: 512px;
                        text-align: right;" class="hd-small">&nbsp; Available sessions:
                        <select class="selectpicker" id="sessionSelect" onchange="selectNewSession()" style="color: #000000;
                            display: inline-block;
                            width: 256px;">
                            <option value="0" selected="selected">Pick a session to track</option>
                        </select>
                    </td>
                    <td style="width: 256px;">&nbsp; <!-- placeholder for future use --></td>
                </tr>
            </tbody>
        </table>

        <br />
        <div style="text-align: center">
            <div id="embed" style="display: inline-block;
                width: 1024px;
                height: 700px;
                border: 1px solid #DEDEDE;"></div>
        </div>
        <br/>
        <br/>
        <div style="text-align: center;
            width: 100%;
            position: fixed;
            bottom: 0px;
            left: 0px;
            margin-bottom: 15px;">
            <div style="display: inline-block;
                margin-right: 48px;">
                <h5>Upload a file to UI server.</h5>
                <form encType="multipart/form-data" action="/tsne/upload" method="POST" id="form">
                    <div>

                        <input name="file" type="file" style="width: 300px;
                            display: inline-block;" />
                        <input type="submit" value="Upload file" style="display: inline-block;"/>

                    </div>
                </form>
            </div>

            """),format.raw/*206.53*/("""
                """),format.raw/*207.96*/("""
                """),format.raw/*208.42*/("""
                    """),format.raw/*209.59*/("""
                    """),format.raw/*210.68*/("""
                """),format.raw/*211.27*/("""
            """),format.raw/*212.23*/("""
        """),format.raw/*213.9*/("""</div>
    </body>

</html>"""))
      }
    }
  }

  def render(): play.twirl.api.HtmlFormat.Appendable = apply()

  def f:(() => play.twirl.api.HtmlFormat.Appendable) = () => apply()

  def ref: this.type = this

}


}

/**/
object Tsne extends Tsne_Scope0.Tsne
              /*
                  -- GENERATED --
                  DATE: Fri May 18 18:41:46 PDT 2018
                  SOURCE: C:/develop/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/tsne/Tsne.scala.html
                  HASH: ac938699c7d11da35fb782a100e2bbf41828e2b5
                  MATRIX: 634->0|1838->1176|1867->1177|1909->1191|2031->1286|2060->1287|2099->1299|2134->1306|2163->1307|2205->1321|2392->1481|2421->1482|2460->1494|2498->1504|2527->1505|2569->1519|2691->1614|2720->1615|2759->1627|2792->1632|2821->1633|2863->1647|3021->1778|3050->1779|3089->1791|3123->1797|3152->1798|3194->1812|3309->1900|3338->1901|3377->1913|3412->1920|3441->1921|3483->1935|3537->1962|3566->1963|3605->1975|3636->1978|3665->1979|3707->1993|3797->2056|3826->2057|3865->2069|3897->2073|3926->2074|3968->2088|4130->2223|4159->2224|4198->2236|4232->2242|4261->2243|4303->2257|4352->2279|4381->2280|4420->2292|4453->2297|4482->2298|4524->2312|4645->2406|4674->2407|4711->2417|4807->2485|4836->2486|4878->2500|4945->2539|4974->2540|5021->2558|5307->2815|5337->2816|5406->2856|5436->2857|5483->2875|5556->2919|5586->2920|5637->2942|5805->3081|5835->3082|5890->3108|6186->3375|6216->3376|6275->3406|6363->3465|6393->3466|6444->3488|6474->3489|6527->3513|6696->3653|6726->3654|6779->3678|6809->3679|6860->3701|6957->3769|6987->3770|7030->3784|7060->3785|7139->3835|7169->3836|7216->3854|7252->3861|7282->3862|7333->3884|7499->4021|7529->4022|7569->4033|7599->4034|7731->4137|7761->4138|7816->4164|7933->4252|7963->4253|8064->4325|8094->4326|8149->4352|8321->4495|8351->4496|8398->4514|8428->4515|8473->4531|8503->4532|8545->4546|8575->4547|10680->6663|10727->6760|10774->6803|10825->6863|10876->6932|10923->6960|10966->6984|11004->6994
                  LINES: 25->1|61->37|61->37|62->38|65->41|65->41|67->43|67->43|67->43|68->44|73->49|73->49|75->51|75->51|75->51|76->52|79->55|79->55|81->57|81->57|81->57|82->58|86->62|86->62|88->64|88->64|88->64|89->65|92->68|92->68|94->70|94->70|94->70|95->71|96->72|96->72|98->74|98->74|98->74|99->75|101->77|101->77|103->79|103->79|103->79|104->80|108->84|108->84|110->86|110->86|110->86|111->87|112->88|112->88|114->90|114->90|114->90|115->91|118->94|118->94|119->95|122->98|122->98|123->99|123->99|123->99|124->100|130->106|130->106|132->108|132->108|133->109|133->109|133->109|134->110|136->112|136->112|137->113|139->115|139->115|140->116|141->117|141->117|142->118|142->118|144->120|148->124|148->124|148->124|148->124|149->125|151->127|151->127|152->128|152->128|155->131|155->131|156->132|156->132|156->132|157->133|160->136|160->136|160->136|160->136|162->138|162->138|163->139|165->141|165->141|166->142|166->142|167->143|170->146|170->146|171->147|171->147|172->148|172->148|175->151|175->151|230->206|231->207|232->208|233->209|234->210|235->211|236->212|237->213
                  -- GENERATED --
              */
          