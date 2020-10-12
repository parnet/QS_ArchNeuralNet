#include "convertwidget.h"
#include "ui_convertwidget.h"

#include <qgpstatistics.h>

ConvertWidget::ConvertWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ConvertWidget)
{
    module = new Module();
    ui->setupUi(this);
    connect(this->ui->btn_loadInput, &QPushButton::clicked, this, &ConvertWidget::loadInputPath);
    connect(this->ui->btn_loadOutput, &QPushButton::clicked, this, &ConvertWidget::loadOutputPath);
}

ConvertWidget::~ConvertWidget()
{
    delete ui;
    delete module;
}

void ConvertWidget::drawMomentum()
{
    ui->bigChart->clearGraphs();
    ui->bigChart->addGraph();
    ui->bigChart->xAxis->setLabel("Bin (log scale)");
    ui->bigChart->yAxis->setLabel("Amount");
    ui->bigChart->xAxis->setRange(0, 21);
    ui->bigChart->replot();

    QGPStatistics stat_nqgp; //  todo = supervisor->getStatistics();
    QGPStatistics stat_qgp;
    size_t *momentum_stat = &stat_qgp.momentum[0];
    size_t *momentum_stat_n = &stat_nqgp.momentum[0];

    QVector<double> x(20), y(20), y_n(20);
    number y_max = 0;
    for (int i = 0; i < 20; i++) x[i] = i;
    for (int i = 0; i < 20; i++) {
        y[i] = momentum_stat[i];
        y_n[i] = momentum_stat_n[i];
        if (y[i] > y_max) y_max = y[i];
        if (y_n[i] > y_max) y_max = y_n[i];
    }
    ui->bigChart->yAxis->setRange(0, y_max + y_max * 0.1);
    ui->bigChart->addGraph();
    ui->bigChart->addGraph();
    ui->bigChart->graph(0)->setData(x, y);
    ui->bigChart->graph(1)->setData(x, y_n);
    QPen pen;
    pen.setColor(QColor(25, 135, 25));
    pen.setWidth(2);
    ui->bigChart->graph(0)->setPen(pen);
    pen.setColor(QColor(135, 25, 25));
    pen.setStyle(Qt::DashLine);
    pen.setWidth(2);
    ui->bigChart->graph(1)->setPen(pen);
    ui->bigChart->replot();
}

void ConvertWidget::drawPolar()
{
    ui->bigChart->clearGraphs();
    ui->bigChart->addGraph();
    ui->bigChart->xAxis->setLabel("Bin");
    ui->bigChart->yAxis->setLabel("Amount");
    ui->bigChart->xAxis->setRange(0, 21);
    ui->bigChart->replot();


    QGPStatistics stats_qgp; // = supervisor->getStatistics();
    QGPStatistics stats_nqgp;

    size_t *teta_stat = &stats_qgp.polar[0];
    size_t *teta_stat_n = &stats_nqgp.polar[0];

    QVector<double> x(20), y(20), y_n(20);
    number y_max = 0;
    for (int i = 0; i < 20; i++) x[i] = i;
    for (int i = 0; i < 20; i++) {
        y[i] = teta_stat[i];
        y_n[i] = teta_stat_n[i];
        if (y[i] > y_max) y_max = y[i];
        if (y_n[i] > y_max) y_max = y_n[i];
    }
    ui->bigChart->yAxis->setRange(0, y_max + y_max * 0.1);
    ui->bigChart->addGraph();
    ui->bigChart->addGraph();
    ui->bigChart->graph(0)->setData(x, y);
    ui->bigChart->graph(1)->setData(x, y_n);
    QPen pen;
    pen.setColor(QColor(25, 135, 25));
    pen.setWidth(2);
    ui->bigChart->graph(0)->setPen(pen);
    pen.setColor(QColor(135, 25, 25));
    pen.setStyle(Qt::DashLine);
    pen.setWidth(2);
    ui->bigChart->graph(1)->setPen(pen);
    ui->bigChart->replot();
}

void ConvertWidget::drawInclination()
{
    ui->bigChart->clearGraphs();
    ui->bigChart->addGraph();
    ui->bigChart->xAxis->setLabel("Bin");
    ui->bigChart->yAxis->setLabel("Amount");
    ui->bigChart->xAxis->setRange(0, 21);
    ui->bigChart->replot();

    QGPStatistics stats_qgp; // = supervisor->getStatistics();
    QGPStatistics stats_nqgp;

    size_t *phi_stat = &stats_qgp.inclination[0];
    size_t *phi_stat_n = &stats_nqgp.inclination[0];

    QVector<double> x(20), y(20), y_n(20);
    double y_max = 0;
    for (int i = 0; i < 20; i++) x[i] = i;
    for (int i = 0; i < 20; i++) {
        y[i] = phi_stat[i];
        y_n[i] = phi_stat_n[i];
        if (y[i] > y_max) y_max = y[i];
        if (y_n[i] > y_max) y_max = y_n[i];
    }
    ui->bigChart->yAxis->setRange(0, y_max + y_max * 0.1);
    ui->bigChart->addGraph();
    ui->bigChart->addGraph();
    ui->bigChart->graph(0)->setData(x, y);
    ui->bigChart->graph(1)->setData(x, y_n);
    QPen pen;
    pen.setColor(QColor(25, 135, 25));
    pen.setWidth(2);
    ui->bigChart->graph(0)->setPen(pen);
    pen.setColor(QColor(135, 25, 25));
    pen.setStyle(Qt::DashLine);
    pen.setWidth(2);
    ui->bigChart->graph(1)->setPen(pen);
    ui->bigChart->replot();
}

void ConvertWidget::loadInputPath(){
    QString fDirname = QFileDialog::getExistingDirectory(this, tr("Open raw input data directory"), QDir::homePath());
    QDir dir(fDirname);
    if(dir.exists()){
        QDir dir_nqgp(fDirname + "/nqgp");
        QDir dir_qgp(fDirname + "/qgp");
        if(dir_nqgp.exists() && dir_qgp.exists()){
            this->converter.inFilepath = fDirname.toStdString();
        } else {
            QMessageBox::information(this, tr("not exists"), tr("Path does not contain qgp and nqgp subfolder"), QMessageBox::Ok);
            }
    }
}

void ConvertWidget::loadOutputPath()
{
    QString fDirname = QFileDialog::getExistingDirectory(this, tr("Open raw input data directory"), QDir::homePath());
    QDir dir(fDirname);
    if(dir.exists()){
        QDir dir_nqgp(fDirname + "/nqgp");
        QDir dir_qgp(fDirname + "/qgp");
        if(dir_nqgp.exists() && dir_qgp.exists()){
            QMessageBox::information(this, tr("not exists"), tr("Path already exists"), QMessageBox::Ok);
            qDebug() << fDirname;

        } else {
            dir.mkdir("nqgp");
            dir.mkdir("qgp");
            }
        this->module->filename.path = fDirname.toStdString();
    }
}
