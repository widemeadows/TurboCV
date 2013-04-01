// Export.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "Export.h"


// This is an example of an exported variable
EXPORT_API int nExport=0;

// This is an example of an exported function.
EXPORT_API int fnExport(void)
{
	return 42;
}

// This is the constructor of a class that has been exported.
// see Export.h for the class definition
CExport::CExport()
{
	return;
}
