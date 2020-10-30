#ifndef TRAININGWIDGET_H
#define TRAININGWIDGET_H

#include <QWidget>
#include <supervisor.h>
#include <thread>

namespace Ui {
class TrainingWidget;
}

class TrainingWidget : public QWidget
{
    Q_OBJECT

private:

    int minchart = 80;
    int maxchart = 105;

    Ui::TrainingWidget *ui;

    QTimer * timer;

    std::thread * process = nullptr;

    QString pathDataset;

    size_t szEpoch;

    Supervisor * supervisor;

public:
    explicit TrainingWidget(QWidget *parent = nullptr);
    ~TrainingWidget();
public slots:
    void loadNeuralNet();

    void saveNeuralNet();

    void exportAccuracyData();

    void setTopology(int number);

    void loadDataset();

    void analyze();

    void setTrainingData(int number);

    void setValidationData(int number);

    void setBatch(int number);

    void setEpoch(int number);

    void setRandom(bool state);

    void drawHistogram(size_t iEp = 0);

    void drawLoss(size_t iEp = 0);

    void renew();

};

#endif // TRAININGWIDGET_H
