#include "Normalization/CLI.h"

int main( int argc, char * argv[])
{
	WSICS::Normalization::CLI cli;
	cli.Execute(argc, argv);
	return 0;
}