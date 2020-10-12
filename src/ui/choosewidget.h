#ifndef CHOOSEWIDGET_H
#define CHOOSEWIDGET_H

#include <QWidget>

namespace Ui {
class ChooseWidget;
}

class ChooseWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ChooseWidget(QWidget *parent = nullptr);
    ~ChooseWidget();

    void setParent(QWidget *parent = nullptr);

private:
    Ui::ChooseWidget *ui;
};

#endif // CHOOSEWIDGET_H
