#include "choosewidget.h"
#include "convertwidget.h"
#include "mainwindow.h"
#include "trainingwidget.h"
#include "ui_mainwindow.h"
#include "validationwidget.h"
#include <QKeySequence>
#include <QAction>
#include <QDebug>
#include <QShortcut>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    QKeySequence newTab = QKeySequence(Qt::CTRL + Qt::Key_T);
    QShortcut *actionNewTab = new QShortcut(newTab,this);
    connect(actionNewTab,SIGNAL(activated()),this,SLOT(createTab()));

    QKeySequence switchTab = QKeySequence(Qt::CTRL + Qt::Key_Tab);
    QShortcut *actionSwitchTab = new QShortcut(switchTab,this);
    connect(actionSwitchTab,SIGNAL(activated()),this,SLOT(nextTab()));

    QKeySequence closeTab = QKeySequence(Qt::CTRL + Qt::Key_X);
    QShortcut *actionCloseTab = new QShortcut(closeTab,this);
    connect(actionCloseTab,SIGNAL(activated()),this,SLOT(closeTab()));

    createTab();
}



MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::createTab(){
    this->ui->tabWidget->addTab(new ChooseWidget(this), "Empty Tab");
    this->ui->tabWidget->setCurrentIndex(this->ui->tabWidget->count()-1);
}

void MainWindow::nextTab(){
    auto sz= this->ui->tabWidget->count();
    auto cI = this->ui->tabWidget->currentIndex();
    auto nI = (cI +1) % sz;
    this->ui->tabWidget->setCurrentIndex(nI);
}

void MainWindow::closeTab(){
    auto sz = this->ui->tabWidget->count();
    if(sz > 1){
        auto cI = this->ui->tabWidget->currentIndex();
        auto current = this->ui->tabWidget->widget(cI);
        this->ui->tabWidget->removeTab(cI);
        current->deleteLater();
    }
}

void MainWindow::changeTabTraining()
{
    auto cI = this->ui->tabWidget->currentIndex();
    auto current = this->ui->tabWidget->widget(cI);
    this->ui->tabWidget->removeTab(cI);
    this->ui->tabWidget->insertTab(cI, new TrainingWidget(),"Training"); // todo number
    current->deleteLater();
    this->ui->tabWidget->setCurrentIndex(cI);
}

void MainWindow::changeTabValidation()
{
    auto cI = this->ui->tabWidget->currentIndex();
    auto current = this->ui->tabWidget->widget(cI);
    this->ui->tabWidget->removeTab(cI);
    this->ui->tabWidget->insertTab(cI,new ValidationWidget(),"Validation"); // todo number
    current->deleteLater();
    this->ui->tabWidget->setCurrentIndex(cI);
}

void MainWindow::changeTabConvert()
{
    auto cI = this->ui->tabWidget->currentIndex();
    auto current = this->ui->tabWidget->widget(cI);
    this->ui->tabWidget->removeTab(cI);
    this->ui->tabWidget->insertTab(cI,new ConvertWidget(),"Convert"); // todo number
    current->deleteLater();
    this->ui->tabWidget->setCurrentIndex(cI);
}
