#include <QCoreApplication>
#include <QDebug>
#include <QTime>
#include <QDebug>
#include <QFile>

#include "NerualNetwork/INerualNetwork.h"

namespace
{
    float sigmoida(float val)
    {
        //--- activation function
       return (1.0 / (1.0 + exp(-val)));
    }
    float sigmoidasDerivate(float val)
    {
        //--- activation function derivative
         return (val * (1.0 - val));
    };

    float* inputs_list(const QStringList &strList)
    {
        float* inputs = (float*) malloc((784)*sizeof(float));
        QString str;
        bool ok=true;
        for (int i = 1; i<strList.size();i++)
        {
            str = strList.at(i);
            inputs[i-1]= ( (str.toFloat(&ok) / 255.0 *0.99)+0.01);
        }
        return inputs;
    }

    float* targets_list(const int &j)
    {
         float* targets = (float*) malloc((10)*sizeof(float));
        for (int i = 0; i<10;i++)
        {
            if(i==j)
            targets[i]=(0.99);
            else
            targets[i]=(0.01);
        }

        return targets;
    }
}



int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    INerualNetwork *nW = new INerualNetwork(sigmoida,sigmoidasDerivate,{784,200,10});
    QStringList wordList;
    bool ok=true;
        QFile f("C:/DATASET/mnist_train.csv");
        if (f.open(QIODevice::ReadOnly))
        {
            int qq=0;
//            while(!f.atEnd())
           while(qq<3000)
            {
                qq++;
                if(qq%100==0)
                    qDebug()<<qq;
                QString data;
                data = f.readLine();
                wordList = data.split(',');
                QString str = wordList.at(0);
                float * tmpIN = inputs_list(wordList);
                float * tmpTAR = targets_list(str.toInt(&ok));

                nW->backPropagate(tmpIN,tmpTAR,0.2);

                delete tmpIN;
                delete tmpTAR;
            }

            f.close();
        }
        QFile f2("C:/DATASET/mnist_test.csv");
        if (f2.open(QIODevice::ReadOnly))
        {
            while(!f2.atEnd())
            {
                QString data;
                data = f2.readLine();
                wordList = data.split(',');
                QString str = wordList.at(0);
                qDebug()<<"__________________";
                qDebug()<<"For number "<<str;
                float * tmpIN = inputs_list(wordList);
                float *out_neurons = nW->feedForwarding(tmpIN);
                for (int i = 0; i < nW->getCountNeuronsOutput(); ++i)
                {
                    qDebug() << i <<   out_neurons[i];
                }
                qDebug()<<"__________________";
                delete tmpIN;
                tmpIN = nullptr;
            }

            f2.close();
        }
         delete nW;
        nW = nullptr;
        qDebug()<<"_______________THE____END_______________";


    return a.exec();
}
