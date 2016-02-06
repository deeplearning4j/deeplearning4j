*> \brief \b SISNAN tests input for NaN.
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at 
*            http://www.netlib.org/lapack/explore-html/ 
*
*> \htmlonly
*> Download SISNAN + dependencies 
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/sisnan.f"> 
*> [TGZ]</a> 
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/sisnan.f"> 
*> [ZIP]</a> 
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/sisnan.f"> 
*> [TXT]</a>
*> \endhtmlonly 
*
*  Definition:
*  ===========
*
*       LOGICAL FUNCTION SISNAN( SIN )
* 
*       .. Scalar Arguments ..
*       REAL               SIN
*       ..
*  
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*> SISNAN returns .TRUE. if its argument is NaN, and .FALSE.
*> otherwise.  To be replaced by the Fortran 2003 intrinsic in the
*> future.
*> \endverbatim
*
*  Arguments:
*  ==========
*
*> \param[in] SIN
*> \verbatim
*>          SIN is REAL
*>          Input to test for NaN.
*> \endverbatim
*
*  Authors:
*  ========
*
*> \author Univ. of Tennessee 
*> \author Univ. of California Berkeley 
*> \author Univ. of Colorado Denver 
*> \author NAG Ltd. 
*
*> \date September 2012
*
*> \ingroup auxOTHERauxiliary
*
*  =====================================================================
      LOGICAL FUNCTION SISNAN( SIN )
*
*  -- LAPACK auxiliary routine (version 3.4.2) --
*  -- LAPACK is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*     September 2012
*
*     .. Scalar Arguments ..
      REAL               SIN
*     ..
*
*  =====================================================================
*
*  .. External Functions ..
      LOGICAL SLAISNAN
      EXTERNAL SLAISNAN
*  ..
*  .. Executable Statements ..
      SISNAN = SLAISNAN(SIN,SIN)
      RETURN
      END
