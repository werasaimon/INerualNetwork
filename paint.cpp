#include "paint.h"
#include "ui_paint.h"

#include <iostream>
#include <QFileDialog>
#include <QDebug>

namespace
{
    float* inputs_list(const QStringList &strList)
    {
        float* inputs = (float*) malloc((784)*sizeof(float));
        QString str;
        bool ok=true;
        for (int i = 1; i<strList.size();i++)
        {
            str = strList.at(i);
            inputs[i-1]= ( (str.toFloat(&ok) / 255.0*0.99))+0.01;
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



paint::paint(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::paint)
{
    ui->setupUi(this);

    //===============================================================//

    ui->customPlot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectAxes |
                                    QCP::iSelectLegend | QCP::iSelectPlottables);

    ui->customPlot->xAxis->setRange(0, 50);
    ui->customPlot->yAxis->setRange(0, 100);
    ui->customPlot->axisRect()->setupFullAxesBox();

    ui->customPlot->plotLayout()->insertRow(0);
    QCPTextElement *title = new QCPTextElement(ui->customPlot, "Backpropagate (training)", QFont("sans", 17, QFont::Bold));
    ui->customPlot->plotLayout()->addElement(0, 0, title);

    ui->customPlot->xAxis->setLabel("x Axis");
    ui->customPlot->yAxis->setLabel("y Axis");
    ui->customPlot->legend->setVisible(true);
    QFont legendFont = font();
    legendFont.setPointSize(10);
    ui->customPlot->legend->setFont(legendFont);
    ui->customPlot->legend->setSelectedFont(legendFont);
    ui->customPlot->legend->setSelectableParts(QCPLegend::spItems); // legend box shall not be selectable, only legend items

    ui->customPlot->xAxis->setBasePen(QPen(Qt::white, 1));
    ui->customPlot->yAxis->setBasePen(QPen(Qt::white, 1));
    ui->customPlot->xAxis->setTickPen(QPen(Qt::white, 1));
    ui->customPlot->yAxis->setTickPen(QPen(Qt::white, 1));
    ui->customPlot->xAxis->setSubTickPen(QPen(Qt::white, 1));
    ui->customPlot->yAxis->setSubTickPen(QPen(Qt::white, 1));
    ui->customPlot->xAxis->setTickLabelColor(Qt::white);
    ui->customPlot->yAxis->setTickLabelColor(Qt::white);
    ui->customPlot->xAxis->grid()->setPen(QPen(QColor(140, 140, 140), 1, Qt::DotLine));
    ui->customPlot->yAxis->grid()->setPen(QPen(QColor(140, 140, 140), 1, Qt::DotLine));
    ui->customPlot->xAxis->grid()->setSubGridPen(QPen(QColor(80, 80, 80), 1, Qt::DotLine));
    ui->customPlot->yAxis->grid()->setSubGridPen(QPen(QColor(80, 80, 80), 1, Qt::DotLine));
    ui->customPlot->xAxis->grid()->setSubGridVisible(true);
    ui->customPlot->yAxis->grid()->setSubGridVisible(true);
    ui->customPlot->xAxis->grid()->setZeroLinePen(Qt::NoPen);
    ui->customPlot->yAxis->grid()->setZeroLinePen(Qt::NoPen);
    ui->customPlot->xAxis->setUpperEnding(QCPLineEnding::esSpikeArrow);
    ui->customPlot->yAxis->setUpperEnding(QCPLineEnding::esSpikeArrow);
    QLinearGradient plotGradient;
    plotGradient.setStart(0, 0);
    plotGradient.setFinalStop(0, 350);
    plotGradient.setColorAt(0, QColor(80, 80, 80));
    plotGradient.setColorAt(1, QColor(50, 50, 50));
    ui->customPlot->setBackground(plotGradient);
    QLinearGradient axisRectGradient;
    axisRectGradient.setStart(0, 0);
    axisRectGradient.setFinalStop(0, 350);
    axisRectGradient.setColorAt(0, QColor(80, 80, 80));
    axisRectGradient.setColorAt(1, QColor(30, 30, 30));
    ui->customPlot->axisRect()->setBackground(axisRectGradient);

    //===============================================================//

    connect(ui->pushButton_clear , SIGNAL (clicked()), this, SLOT (push_button_clear()));
    connect(ui->pushButton_load , SIGNAL (clicked()), this, SLOT (push_button_load()));
    connect(ui->pushButton_RandomWeights , SIGNAL (clicked()), this, SLOT (push_button_init_random_weigths()));

    image_size   = {28, 28, QImage::Format_RGB32};
    image_size.fill(Qt::black);

    auto copy = image_size.scaled(w*scale, h*scale);
    ui->label->setPixmap(QPixmap::fromImage(copy));

    NeuralNet = new INerualNetwork({784, 200, 10});

}

paint::~paint()
{
    delete ui;
}



void paint::push_button_clear()
{
    image_size.fill(Qt::black);
    auto copy = image_size.scaled(w*scale, h*scale);
    ui->label->setPixmap(QPixmap::fromImage(copy));
}

void paint::push_button_load()
{
    QString filePath = QFileDialog::getOpenFileName(
                                this,
                                "Open",
                                "",
                                tr("Text Files (*.csv)") );

    qDebug() << filePath;

    QStringList wordList;
    bool ok=true;
    QFile f(filePath);
    if (f.open(QIODevice::ReadOnly))
    {
        float errorSum = 0;
        int right = 0;
        int qq=0;
        int i = 0;


        int max_size = 9000;
        QVector<double> _x_error(max_size);
        QVector<double> _y_error(max_size);
        QVector<double> _x_correct(max_size);
        QVector<double> _y_correct(max_size);

        ui->customPlot->xAxis->setRange(0, 10);
        ui->customPlot->yAxis->setRange(0, 100);

        auto graph1 = AddGraph("error" , _x_error   , _y_error);
        auto graph2 = AddGraph("narma" , _x_correct , _y_correct);

        QPen graphPenError;
        graphPenError.setColor(QColor(255,0,0));
        graphPenError.setWidthF(1);
        graph1->setPen(graphPenError);

        QPen graphPenNorm;
        graphPenNorm.setColor(QColor(0,255,0));
        graphPenNorm.setWidthF(1);
        graph2->setPen(graphPenNorm);

        while(!f.atEnd())
        //while(qq < max_size)
        {
            qq++;
            if(qq%100==0)
            {
                qDebug()<<qq;
                qDebug() << "epoch: " << (i++) << " correct: " << right << " error: " << errorSum;

                ui->customPlot->xAxis->setRange(0, 50 + (((i-10) > 0) ? i-10 : 0) );

                _x_error[i] = i;
                _x_correct[i] = i;

                _y_error[i] = errorSum;
                _y_correct[i] = right;

                graph1->setData(_x_error, _y_error);
                graph2->setData(_x_correct, _y_correct);
                ui->customPlot->replot();

                errorSum = 0;
                right = 0;

            }

            QString data;
            data = f.readLine();
            wordList = data.split(',');
            QString str = wordList.at(0);
            float * tmpIN = inputs_list(wordList);
            int digit = str.toInt(&ok);
            float * tmpTAR = targets_list(digit);


            float *outputs =  NeuralNet->feedForwarding(tmpIN);

            int maxDigit = 0;
            float maxDigitWeight = -1;
            for (int k = 0; k < 10; k++)
            {
                if(outputs[k] > maxDigitWeight)
                {
                    maxDigitWeight = outputs[k];
                    maxDigit = k;
                }
            }

            if(digit == maxDigit) right++;
            for (int k = 0; k < 10; k++)
            {
                errorSum += (tmpTAR[k] - outputs[k]) * (tmpTAR[k] - outputs[k]);
            }

            NeuralNet->backPropagate(tmpIN,tmpTAR,ui->doubleSpinBox_LerningRate->value());

            delete tmpIN;
            delete tmpTAR;
        }

        f.close();
    }
}

void paint::push_button_init_random_weigths()
{
    if(NeuralNet)
    {
        ui->customPlot->clearGraphs();
        ui->customPlot->replot();
        NeuralNet->InitRandom();
    }
}


//-------------------------------------//

void paint::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
}

void paint::mousePressEvent(QMouseEvent *e)
{
    int key = 0;
    if(e->buttons() == Qt::LeftButton) key = 1;
    if(e->buttons() == Qt::RightButton) key = 2;
    mx = e->x()/scale;
    my = e->y()/scale;
    mousePressed=key;

}

void paint::mouseMoveEvent(QMouseEvent *e)
{
    int key = 0;
    if(e->buttons() == Qt::LeftButton) key = 1;
    if(e->buttons() == Qt::RightButton) key = 2;
    mx = e->x()/scale;
    my = e->y()/scale;
    mousePressed=key;


    for (int i = 0; i < w; i++)
    {
        for (int j = 0; j < h; j++)
        {
            auto p = image_size.pixel(i,j);
            //int colors= qRed(p) | qGreen(p) | qBlue(p);
            int colors= qBlue(p);
            if(mousePressed != 0)
            {
                double dist = ((i - mx) * (i - mx) +
                               (j - my) * (j - my)) * 3 + 0.5;// * 3;// / 28.0;

                if(dist < 1) dist = 1; else dist *= dist;

                if(mousePressed == 1)
                {
                    colors += 255 / dist;
                }
                else if(mousePressed == 2 && dist < 10)
                {
                    colors -= 255 / dist;
                }

                colors = std::max(0,std::min(colors,255));
            }
            // int color = (int)(colors) * 255;
            //int col = (colors << 16) | (colors << 8) | colors;
            image_size.setPixel(i, j, colors);
            inputs[i + j * w] = qBlue(colors) / 255.0;
        }
    }

    float *output = NeuralNet->feedForwarding(inputs);


    ui->progressBar_0->setValue(output[0] * 100.0);
    ui->progressBar_1->setValue(output[1] * 100.0);
    ui->progressBar_2->setValue(output[2] * 100.0);
    ui->progressBar_3->setValue(output[3] * 100.0);
    ui->progressBar_4->setValue(output[4] * 100.0);
    ui->progressBar_5->setValue(output[5] * 100.0);
    ui->progressBar_6->setValue(output[6] * 100.0);
    ui->progressBar_7->setValue(output[7] * 100.0);
    ui->progressBar_8->setValue(output[8] * 100.0);
    ui->progressBar_9->setValue(output[9] * 100.0);



    int maxDigit = 0;
    float maxDigitWeight = -1;
    for (int k = 0; k < 10; k++)
    {
        if(output[k] > maxDigitWeight)
        {
            maxDigitWeight = output[k];
            maxDigit = k;
        }
    }

    ui->label_digit->setText(QString(" Hаверное это:  ") + QString::number(maxDigit) + QString("  "));

    int _w = ui->label->width();
    int _h = ui->label->height();
    ui->label->setPixmap(QPixmap::fromImage(image_size.scaled(_w, _h)));
}

QCPGraph *paint::AddGraph(QString func_name, QVector<double> _x, QVector<double> _y)
{
    auto graph1 = ui->customPlot->addGraph();
    graph1->setName( func_name );// + QString(" graph %1").arg(ui->customPlot->graphCount()-1));
    graph1->setData(_x, _y);
    graph1->setLineStyle((QCPGraph::LineStyle)(1));
    graph1->setScatterStyle(QCPScatterStyle((QCPScatterStyle::ScatterShape::ssPlus)));

    QPen graphPen;
    graphPen.setColor(QColor(rand()%245+10, rand()%245+10, rand()%245+10));
    graphPen.setWidthF(1);
    graph1->setPen(graphPen);
    ui->customPlot->replot();

    return graph1;
}


void paint::on_actionsave_triggered()
{
    QString filter = "Weigths Files (*.w)";
    QString filename = QFileDialog::getSaveFileName(this, "Save file", "", filter , &filter);
    NeuralNet->save(filename.toStdString());

}

void paint::on_actionload_triggered()
{
    QString filter = "Weigths Files (*.w)";
    QString filename = QFileDialog::getOpenFileName(this, "Open file", "",  filter , &filter);
    if(NeuralNet)
    {
        delete NeuralNet;
    }
    NeuralNet = NeuralNet->load(filename.toStdString());
}
