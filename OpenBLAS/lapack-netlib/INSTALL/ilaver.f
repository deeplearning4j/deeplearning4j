*> \brief \b ILAVER returns the LAPACK version.
**
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at 
*            http://www.netlib.org/lapack/explore-html/ 
*
*  Definition:
*  ===========
*
*     SUBROUTINE ILAVER( VERS_MAJOR, VERS_MINOR, VERS_PATCH )
*
*     INTEGER VERS_MAJOR, VERS_MINOR, VERS_PATCH
*  
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*>  This subroutine returns the LAPACK version.
*> \endverbatim
*
*  Arguments:
*  ==========
*
*>  \param[out] VERS_MAJOR
*>      return the lapack major version
*>
*>  \param[out] VERS_MINOR
*>      return the lapack minor version from the major version
*>
*>  \param[out] VERS_PATCH
*>      return the lapack patch version from the minor version
*
*  Authors:
*  ========
*
*> \author Univ. of Tennessee 
*> \author Univ. of California Berkeley 
*> \author Univ. of Colorado Denver 
*> \author NAG Ltd. 
*
*> \date November 2015
*
*> \ingroup auxOTHERauxiliary
*
*  =====================================================================
      SUBROUTINE ILAVER( VERS_MAJOR, VERS_MINOR, VERS_PATCH )
*
*  -- LAPACK computational routine (version 3.6.0) --
*  -- LAPACK is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*     November 2015
*
*  =====================================================================
*
      INTEGER VERS_MAJOR, VERS_MINOR, VERS_PATCH
*  =====================================================================
      VERS_MAJOR = 3
      VERS_MINOR = 6
      VERS_PATCH = 0
*  =====================================================================
*
      RETURN
      END
