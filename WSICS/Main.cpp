#include "Standardization/SlideStandardizationCLI.h"

int main( int argc, char * argv[])
{
	WSICS::Standardization::SlideStandardizationCLI cli;
	cli.Execute(argc, argv);
	return 0;
}