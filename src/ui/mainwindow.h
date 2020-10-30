#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <plotwidget.h>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

public slots:
    void createTab();
    void nextTab();
    void closeTab();
    void changeTabTraining();
    void changeTabValidation();
    void changeTabConvert();
    void changeTabPlotting();
private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
