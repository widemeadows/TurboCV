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

                    ////��ø�Ԫ�أ���Persons��
                    //TiXmlElement *RootElement = myDocument->RootElement();
                    ////�����Ԫ�����ƣ������Persons��
                    //cout << RootElement->Value() << endl;
                    ////��õ�һ��Person�ڵ㡣
                    //TiXmlElement *FirstPerson = RootElement->FirstChildElement();
                    ////��õ�һ��Person��name�ڵ��age�ڵ��ID���ԡ�
                    //TiXmlElement *NameElement = FirstPerson->FirstChildElement();
                    //TiXmlElement *AgeElement = NameElement->NextSiblingElement();
                    //TiXmlAttribute *IDAttribute = FirstPerson->FirstAttribute();
                    ////�����һ��Person��name���ݣ��������ǣ�age���ݣ�����ID���ԣ�����
                }
                catch (string& e) {}
            }

        protected:
            map<String, double> params;
        };
    }
}
}