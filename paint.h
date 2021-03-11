#ifndef PAINT_H
#define PAINT_H

#include <QMainWindow>

#include <QWidget>
#include <QTimer>
#include <QResizeEvent>

#include "NerualNetwork/INerualNetwork.h"
#include "qcustomplot.h"

QT_BEGIN_NAMESPACE
namespace Ui { class paint; }
QT_END_NAMESPACE

class paint : public QMainWindow
{
    Q_OBJECT

    int size = 1;
    int w = 28*size;
    int h = 28*size;
    int scale = 20/size;

public:
    paint(QWidget *parent = nullptr);
    ~paint();

private:

    Ui::paint *ui;

    QImage image_size;
    INerualNetwork *NeuralNet;

    float inputs[28*28];
    float result[10];

    float mx,my;
    int mousePressed;


private:
    /* Переопределяем событие изменения размера окна
     * для пересчёта размеров графической сцены
     * */
    void resizeEvent(QResizeEvent * event);

    // Для рисования используем события мыши
    void mousePressEvent(QMouseEvent *e);
    void mouseMoveEvent(QMouseEvent *e);


    QCPGraph *AddGraph(QString func_name, QVector<double> _x , QVector<double> _y );

private slots:

    void push_button_clear();
    void push_button_load();
    void push_button_init_random_weigths();



    void on_actionsave_triggered();
    void on_actionload_triggered();
};
#endif // PAINT_H
