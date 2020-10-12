#ifndef LOGGING_H
#define LOGGING_H

#include "environment.h"

void loggingFileQT(QtMsgType type, const QMessageLogContext &context , const QString & msg){
    QString txt = msg;
    QFile outFile("M:/log.txt");
    outFile.open(QIODevice::WriteOnly | QIODevice::Append);
    QTextStream ts(&outFile);
    ts << txt << endl; // to file
    QTextStream(stdout)<< txt <<endl; // to console
}

void loggingFile(QtMsgType type, const QMessageLogContext &context , const QString & msg){
    QString txt = msg;
    QFile outFile("M:/log.txt");
    outFile.open(QIODevice::WriteOnly | QIODevice::Append);
    QTextStream ts(&outFile);
    ts << txt << endl; // to file
    //QTextStream(stdout)<< txt <<endl; // to console
}
#endif // LOGGING_H
