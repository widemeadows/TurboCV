#include "../System/System.h"
#include "../System.XML/System.XML.h"
#include "Solver.hpp"
#include <string>
#include <map>
#include <algorithm>
using namespace TurboCV::System::XML;
using namespace std;

namespace TurboCV
{
namespace System
{
    namespace Image
    {
        //////////////////////////////////////////////////////////////////////////
        // Solver
        //////////////////////////////////////////////////////////////////////////

        TiXmlDocument Solver::LoadConfiguration(const String& configfileName)
        {
            TiXmlDocument doc((const char*)configfileName);
            doc.LoadFile();

            return doc;
        }

        Group<ArrayList<String>, ArrayList<int>> Solver::LoadDataset(const String& datasetPath)
        {
            DirectoryInfo imageSetInfo(datasetPath);

            ArrayList<String> classInfos = imageSetInfo.GetDirectories();
            sort(classInfos.begin(), classInfos.end());

            ArrayList<String> imagePaths;
            ArrayList<int> imageLabels;
            for (int i = 0; i < classInfos.Count(); i++)
            {
                ArrayList<String> fileInfos = DirectoryInfo(classInfos[i]).GetFiles();
                sort(fileInfos.begin(), fileInfos.end());

                for (int j = 0; j < fileInfos.Count(); j++)
                {
                    imagePaths.Add(fileInfos[j]);
                    imageLabels.Add(i + 1);
                }
            }

            return CreateGroup(imagePaths, imageLabels);
        }

        map<String, String> Solver::GetConfiguration(const String& featureName)
        {
            map<String, String> params;

            try
            {
                TiXmlNode* configs = doc.RootElement()->FirstChild();
                TiXmlElement* config = configs->FirstChild((const char*)featureName)->ToElement();

                TiXmlAttribute* attr = config->FirstAttribute();
                params[attr->Name()] = attr->Value();

                while ((attr = attr->Next()) != NULL)
                    params[attr->Name()] = attr->Value();
            }
            catch (string& e) 
            {
                return map<String, String>();
            }

            return params;
        }
    }
}
}