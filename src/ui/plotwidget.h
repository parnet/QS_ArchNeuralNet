#ifndef PLOTWIDGET_H
#define PLOTWIDGET_H

#include <QWidget>
#include <supervisor.h>
#include <fstream>
#include <vector>
#include <QVector>
#include <qcustomplot.h>

// TODO: implement variables for graph styles, color preview, make graphs (in-)visible on demand.

namespace Ui {
class PlotWidget;
}

class PlotWidget : public QWidget
{
    Q_OBJECT

private:
    Ui::PlotWidget *ui;
    Supervisor *supervisor;
    std::vector<std::vector<double>> totalPlotData;
    int yMinRange = 50;
    int yMaxRange = 105;
    int xMaxRange = 1;
    int nGraphs = 0;

    std::vector<int> color1 = {255, 125, 125};
    std::vector<int> color2 = {125, 125, 255};
    QColor defaultTGraphColor = QColor(255, 125, 125);
    QColor defaultVGraphColor = QColor(125, 125, 255);

    QString tGraphName = "Training Graph";
    QString vGraphName = "Validation Graph";

    QCPScatterStyle tGraphStyle = QCPScatterStyle::ssCross;
    QCPScatterStyle vGraphStyle = QCPScatterStyle::ssPlus;


public:
    explicit PlotWidget(QWidget *parent = nullptr);
    ~PlotWidget();

public slots:
    void addPlotData();
    void resetPlotData();

    void changeColorTrainingR(int value);
    void changeColorTrainingG(int value);
    void changeColorTrainingB(int value);
    void changeColorValidationR(int value);
    void changeColorValidationG(int value);
    void changeColorValidationB(int value);
    void changeYmax(int value);
    void changeYmin(int value);
    void changeTGraphDesign(int value);
    void changeVGraphDesign(int value);
    void setTGraphName(QString str);
    void setVGraphName(QString str);
    void exportToPNG();


};

#endif // PLOTWIDGET_H
