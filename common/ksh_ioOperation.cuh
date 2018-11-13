#include <ksh_common.cuh>
#include <sstream>
#include <string.h>
#include <string>
#include <stdio.h>
#include <time.h>
#include <ctime>
#include <fstream>
#include <stdio.h>
#include <time.h>
#include <iostream>

using namespace std;


bool openFile(ofstream &fout, const char* fileNamePrefix, const char* fileName, const char* fileExtension, std::ios_base::openmode mode)
{ 
	//ofstream fout;
	stringstream ss;
	if (fileName)
		ss << fileNamePrefix << fileName << "." << fileExtension;
	else
		ss << fileNamePrefix << "." << fileExtension;
	//printf("fileName: %s\n", ss.str().c_str());
	const char* f = ss.str().c_str();
	fout.open(f, std::ios_base::out);
	
	return true;
}
 
bool openFileOut(ofstream &fout, const char* fileNamePrefix, const char* fileName, const char* fileExtension)
{ 
	return openFile(fout, fileNamePrefix, fileName, fileExtension, std::ios_base::out);
}

bool openFileOAppend(ofstream &fout, const char* fileNamePrefix, const char* fileName, const char* fileExtension)
{ 
	return openFile(fout, fileNamePrefix, fileName, fileExtension, std::ios_base::app);
}

bool openFileOut(ofstream &fout, string fileNamePrefix, string fileName, string fileExtension)
{ 
	return openFile(fout, fileNamePrefix.c_str(), fileName.c_str(), fileExtension.c_str(), std::ios_base::out);
}

bool openFileOAppend(ofstream &fout, string fileNamePrefix, string fileName, string fileExtension)
{ 
	return openFile(fout, fileNamePrefix.c_str(), fileName.c_str(), fileExtension.c_str(), std::ios_base::app);
}

string getCommandLineParams(int argc, char **argv)
{
	if (!argc)
		return NULL;
	stringstream ss;
	for (int i = 0;i < argc;i++)
		ss << argv[i] << " ";
	return ss.str();
}