#include "../System/System.h"
#include "../System.XML/System.XML.h"
#include <string>
#include <map>
using namespace TurboCV::System::XML;
using namespace std;

namespace TurboCV
{
namespace System
{
    namespace Image
    {
        class Solver
        {
        public:
            Solver(const String& configFilePath)
            {
                try
                {
                    TiXmlDocument* doc = new TiXmlDocument(configFilePath);
                    doc->LoadFile();

                    //TiXmlElement* root = doc->RootElement();
                    //root->

                    ////获得根元素，即Persons。
                    //TiXmlElement *RootElement = myDocument->RootElement();
                    ////输出根元素名称，即输出Persons。
                    //cout << RootElement->Value() << endl;
                    ////获得第一个Person节点。
                    //TiXmlElement *FirstPerson = RootElement->FirstChildElement();
                    ////获得第一个Person的name节点和age节点和ID属性。
                    //TiXmlElement *NameElement = FirstPerson->FirstChildElement();
                    //TiXmlElement *AgeElement = NameElement->NextSiblingElement();
                    //TiXmlAttribute *IDAttribute = FirstPerson->FirstAttribute();
                    ////输出第一个Person的name内容，即周星星；age内容，即；ID属性，即。
                }
                catch (string& e) {}
            }

        protected:
            map<String, double> params;
        };
    }
}
}